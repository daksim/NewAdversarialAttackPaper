# Latest Large Language Model Attack Papers
**update at 2024-10-14 09:43:09**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On the Adversarial Transferability of Generalized "Skip Connections"**

广义“跳过连接”的对抗性可转让性 cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08950v1) [paper-pdf](http://arxiv.org/pdf/2410.08950v1)

**Authors**: Yisen Wang, Yichuan Mo, Dongxian Wu, Mingjie Li, Xingjun Ma, Zhouchen Lin

**Abstract**: Skip connection is an essential ingredient for modern deep models to be deeper and more powerful. Despite their huge success in normal scenarios (state-of-the-art classification performance on natural examples), we investigate and identify an interesting property of skip connections under adversarial scenarios, namely, the use of skip connections allows easier generation of highly transferable adversarial examples. Specifically, in ResNet-like models (with skip connections), we find that using more gradients from the skip connections rather than the residual modules according to a decay factor during backpropagation allows one to craft adversarial examples with high transferability. The above method is termed as Skip Gradient Method (SGM). Although starting from ResNet-like models in vision domains, we further extend SGM to more advanced architectures, including Vision Transformers (ViTs) and models with length-varying paths and other domains, i.e. natural language processing. We conduct comprehensive transfer attacks against various models including ResNets, Transformers, Inceptions, Neural Architecture Search, and Large Language Models (LLMs). We show that employing SGM can greatly improve the transferability of crafted attacks in almost all cases. Furthermore, considering the big complexity for practical use, we further demonstrate that SGM can even improve the transferability on ensembles of models or targeted attacks and the stealthiness against current defenses. At last, we provide theoretical explanations and empirical insights on how SGM works. Our findings not only motivate new adversarial research into the architectural characteristics of models but also open up further challenges for secure model architecture design. Our code is available at https://github.com/mo666666/SGM.

摘要: 跳过连接是现代深层模型更深入、更强大的关键因素。尽管它们在正常场景中取得了巨大的成功(在自然示例上的最新分类性能)，但我们调查并识别了对抗性场景下跳过连接的一个有趣属性，即使用跳过连接可以更容易地生成高度可转移的对抗性示例。具体地说，在类ResNet模型(带有跳过连接)中，我们发现在反向传播过程中，根据衰减因子使用来自跳过连接的更多梯度，而不是使用剩余模块，可以创建具有高可转移性的对抗性例子。上述方法被称为跳过梯度法(SGM)。虽然我们从视觉领域中类似ResNet的模型开始，但我们将SGM进一步扩展到更高级的体系结构，包括视觉转换器(VITS)和具有变长度路径的模型以及其他领域，即自然语言处理。我们针对不同的模型进行全面的传输攻击，包括ResNet、Transformers、Inceptions、Neural Architecture Search和Large Language Model(LLM)。我们表明，在几乎所有情况下，使用SGM都可以极大地提高精心设计的攻击的可转移性。此外，考虑到实际应用的巨大复杂性，我们进一步证明了SGM甚至可以提高模型集成或定向攻击的可转换性和对现有防御的隐蔽性。最后，本文对SGM的运行机制进行了理论解释和实证分析。我们的发现不仅激发了对模型体系结构特征的新的对抗性研究，而且也为安全模型体系结构设计开辟了进一步的挑战。我们的代码可以在https://github.com/mo666666/SGM.上找到



## **2. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08827v1) [paper-pdf](http://arxiv.org/pdf/2410.08827v1)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **3. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

PoisonBench：评估大型语言模型数据中毒漏洞 cs.CR

Tingchen Fu and Fazl Barez are core research contributors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08811v1) [paper-pdf](http://arxiv.org/pdf/2410.08811v1)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.

摘要: 偏好学习是调整当前LLM的核心组件，但此过程很容易受到数据中毒攻击。为了解决这一问题，我们引入了PoisonBch，这是一个评估大型语言模型在偏好学习过程中对数据中毒敏感性的基准。数据中毒攻击可以操纵大型语言模型响应，以包括隐藏的恶意内容或偏见，从而可能导致模型在看起来正常运行的情况下生成有害或意外的输出。我们在八个现实场景中部署了两种不同的攻击类型，评估了21个广泛使用的模型。我们的发现揭示了以下趋势：(1)增大参数大小并不能本质上增强对中毒攻击的抵御能力；(2)攻击效果与数据毒化比率之间存在对数线性关系；(3)数据中毒的影响可以推广到中毒数据中没有包括的外推触发器。这些结果暴露了当前偏好学习技术的弱点，突显出迫切需要更强大的防御恶意模型和数据操纵。



## **4. F2A: An Innovative Approach for Prompt Injection by Utilizing Feign Security Detection Agents**

F2A：利用Feign安全检测代理进行即时注入的创新方法 cs.CR

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08776v1) [paper-pdf](http://arxiv.org/pdf/2410.08776v1)

**Authors**: Yupeng Ren

**Abstract**: With the rapid development of Large Language Models (LLMs), numerous mature applications of LLMs have emerged in the field of content safety detection. However, we have found that LLMs exhibit blind trust in safety detection agents. The general LLMs can be compromised by hackers with this vulnerability. Hence, this paper proposed an attack named Feign Agent Attack (F2A).Through such malicious forgery methods, adding fake safety detection results into the prompt, the defense mechanism of LLMs can be bypassed, thereby obtaining harmful content and hijacking the normal conversation.Continually, a series of experiments were conducted. In these experiments, the hijacking capability of F2A on LLMs was analyzed and demonstrated, exploring the fundamental reasons why LLMs blindly trust safety detection results. The experiments involved various scenarios where fake safety detection results were injected into prompts, and the responses were closely monitored to understand the extent of the vulnerability. Also, this paper provided a reasonable solution to this attack, emphasizing that it is important for LLMs to critically evaluate the results of augmented agents to prevent the generating harmful content. By doing so, the reliability and security can be significantly improved, protecting the LLMs from F2A.

摘要: 随着大语言模型的快速发展，大语言模型在内容安全检测领域出现了大量成熟的应用。然而，我们发现LLM在安全检测代理中表现出盲目信任。一般的LLMS可能会被黑客利用此漏洞攻击。为此，提出了一种伪装代理攻击(F2A)，通过这种恶意伪造方法，将虚假的安全检测结果添加到提示中，绕过LLMS的防御机制，从而获取有害内容，劫持正常会话。在这些实验中，分析和论证了F2A对LLMS的劫持能力，探索了LLMS盲目相信安全检测结果的根本原因。这些实验涉及各种场景，在提示中注入虚假的安全检测结果，并密切监控响应，以了解漏洞的程度。此外，本文还提供了一种合理的解决方案，强调了对于LLMS来说，批判性地评估增强剂的结果对于防止产生有害内容是很重要的。通过这样做，可以显著提高可靠性和安全性，保护LLMS免受F2A的影响。



## **5. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

RePD：通过基于检索的即时分解过程防御越狱攻击 cs.CR

arXiv admin note: text overlap with arXiv:2403.04783 by other authors

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2410.08660v1) [paper-pdf](http://arxiv.org/pdf/2410.08660v1)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.

摘要: 在这项研究中，我们介绍了RePD，一个创新的基于攻击检索的提示分解框架，旨在降低对大型语言模型(LLM)的越狱攻击风险。尽管严格的预训和微调侧重于道德一致性，但LLM仍然容易受到越狱利用的影响。RePD运行在一次性学习模式上，其中它访问预先收集的越狱提示模板数据库，以识别和分解嵌入用户提示中的有害查询。这一过程包括将越狱提示的分解集成到用户的原始查询中，并将其整合为一个一次性学习示例，以有效地教会LLM识别和分离恶意组件。因此，LLM配备了首先中和任何潜在有害元素，然后以符合其道德准则的方式处理用户的提示。RePD是通用的，并与各种作为代理的开源LLM兼容。通过对有害提示和良性提示的全面实验，我们已经证明了我们提出的RePD在增强LLM对越狱攻击的弹性方面的有效性，而不会影响它们响应典型用户请求的性能。



## **6. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

ART：文本到图像模型的自动红色团队以保护良性用户 cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2405.19360v3) [paper-pdf](http://arxiv.org/pdf/2405.19360v3)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.

摘要: 由于具有创造内容的能力，大规模的预先训练的生成性模型正在席卷世界。同时，为了保护用户的权利和安全，制定了对这些生成模型的保障措施，其中大部分是为大型语言模型设计的。现有的方法主要针对越狱和对抗性攻击，主要是在恶意提示下对模型的安全性进行评估。最近的研究发现，手动创建的安全提示可能会无意中引发不安全的世代。为了进一步系统地评估文本到图像模型的安全风险，我们提出了一个新的自动红色团队框架ART。我们的方法利用视觉语言模型和大型语言模型来建立不安全生成及其提示之间的联系，从而更有效地识别模型的漏洞。通过我们的综合实验，我们揭示了流行的开源文本到图像模型的毒性。实验也验证了ART的有效性、适应性和多样性。此外，我们还介绍了三个大型红团队数据集，用于研究与文本到图像模型相关的安全风险。数据集和模型可在https://github.com/GuanlinLee/ART.中找到



## **7. Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models**

多模式大型语言模型中检测越狱的跨模式信息检查 cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-10-11    [abs](http://arxiv.org/abs/2407.21659v3) [paper-pdf](http://arxiv.org/pdf/2407.21659v3)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both white-box and black-box MLLMs.

摘要: 多通道大语言模型扩展了多通道大语言模型对多通道信息的理解能力，在许多以视觉为中心的任务中取得了显著的性能。尽管如此，最近的研究表明，这些模型容易受到越狱攻击，越狱攻击指的是一种利用技术，恶意用户可以破坏目标模型的安全对齐，并生成误导性和有害的答案。这种潜在的威胁既是由LLM固有的漏洞造成的，也是由视觉输入引入的更大的攻击范围造成的。为了提高MLMS抵御越狱攻击的安全性，研究人员开发了各种防御技术。然而，这些方法要么需要修改模型的内部结构，要么在推理阶段需要大量的计算资源。多式联运信息是一把双刃剑。虽然它增加了攻击的风险，但它也提供了额外的数据，可以加强安全措施。受此启发，我们提出了跨模式信息检测器(Cider)，这是一种即插即用的越狱检测器，旨在利用有害查询和敌意图像之间的跨模式相似性来识别恶意扰动的图像输入。苹果酒不依赖于目标MLLM，并且需要较少的计算成本。大量的实验结果证明了苹果酒的有效性和效率，以及它对白盒和黑盒MLLMS的可转换性。



## **8. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06186v2) [paper-pdf](http://arxiv.org/pdf/2410.06186v2)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **9. APOLLO: A GPT-based tool to detect phishing emails and generate explanations that warn users**

APOLLO：一个基于GPT的工具，用于检测网络钓鱼电子邮件并生成警告用户的解释 cs.HC

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07997v1) [paper-pdf](http://arxiv.org/pdf/2410.07997v1)

**Authors**: Giuseppe Desolda, Francesco Greco, Luca Viganò

**Abstract**: Phishing is one of the most prolific cybercriminal activities, with attacks becoming increasingly sophisticated. It is, therefore, imperative to explore novel technologies to improve user protection across both technical and human dimensions. Large Language Models (LLMs) offer significant promise for text processing in various domains, but their use for defense against phishing attacks still remains scarcely explored. In this paper, we present APOLLO, a tool based on OpenAI's GPT-4o to detect phishing emails and generate explanation messages to users about why a specific email is dangerous, thus improving their decision-making capabilities. We have evaluated the performance of APOLLO in classifying phishing emails; the results show that the LLM models have exemplary capabilities in classifying phishing emails (97 percent accuracy in the case of GPT-4o) and that this performance can be further improved by integrating data from third-party services, resulting in a near-perfect classification rate (99 percent accuracy). To assess the perception of the explanations generated by this tool, we also conducted a study with 20 participants, comparing four different explanations presented as phishing warnings. We compared the LLM-generated explanations to four baselines: a manually crafted warning, and warnings from Chrome, Firefox, and Edge browsers. The results show that not only the LLM-generated explanations were perceived as high quality, but also that they can be more understandable, interesting, and trustworthy than the baselines. These findings suggest that using LLMs as a defense against phishing is a very promising approach, with APOLLO representing a proof of concept in this research direction.

摘要: 网络钓鱼是最频繁的网络犯罪活动之一，攻击变得越来越复杂。因此，必须探索新技术，从技术和人力两个层面改善对用户的保护。大型语言模型(LLM)为各个领域的文本处理提供了巨大的希望，但它们用于防御网络钓鱼攻击的研究仍然很少。在本文中，我们提出了一个基于OpenAI的GPT-4o的工具Apollo，它可以检测钓鱼电子邮件，并向用户生成解释消息，说明为什么特定的电子邮件是危险的，从而提高他们的决策能力。我们评估了Apollo在分类钓鱼电子邮件方面的性能；结果表明，LLM模型在分类钓鱼电子邮件方面具有典范的能力(在GPT-40的情况下准确率为97%)，并且通过整合来自第三方服务的数据可以进一步提高这一性能，从而产生近乎完美的分类率(99%的准确率)。为了评估人们对该工具产生的解释的看法，我们还对20名参与者进行了一项研究，比较了四种不同的解释作为网络钓鱼警告。我们将LLM生成的解释与四个基线进行了比较：手动创建的警告，以及来自Chrome、Firefox和Edge浏览器的警告。结果表明，LLM生成的解释不仅被认为是高质量的，而且比基线更容易理解、更有趣、更可信。这些发现表明，使用LLMS作为对网络钓鱼的防御是一种非常有前途的方法，阿波罗代表了这一研究方向的概念证明。



## **10. Towards Assurance of LLM Adversarial Robustness using Ontology-Driven Argumentation**

使用实体驱动论证确保LLM对抗鲁棒性 cs.AI

To be published in xAI 2024, late-breaking track

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07962v1) [paper-pdf](http://arxiv.org/pdf/2410.07962v1)

**Authors**: Tomas Bueno Momcilovic, Beat Buesser, Giulio Zizzo, Mark Purcell, Dian Balta

**Abstract**: Despite the impressive adaptability of large language models (LLMs), challenges remain in ensuring their security, transparency, and interpretability. Given their susceptibility to adversarial attacks, LLMs need to be defended with an evolving combination of adversarial training and guardrails. However, managing the implicit and heterogeneous knowledge for continuously assuring robustness is difficult. We introduce a novel approach for assurance of the adversarial robustness of LLMs based on formal argumentation. Using ontologies for formalization, we structure state-of-the-art attacks and defenses, facilitating the creation of a human-readable assurance case, and a machine-readable representation. We demonstrate its application with examples in English language and code translation tasks, and provide implications for theory and practice, by targeting engineers, data scientists, users, and auditors.

摘要: 尽管大型语言模型（LLM）具有令人印象深刻的适应性，但在确保其安全性、透明度和可解释性方面仍然存在挑战。鉴于LLM容易受到对抗攻击，需要通过对抗训练和护栏的不断发展的组合来保护它们。然而，管理隐性和异类知识以持续确保稳健性是困难的。我们引入了一种新颖的方法来确保LLM的对抗稳健性，基于正式论证。使用实体进行形式化，我们构建了最先进的攻击和防御，促进了人类可读的保证案例和机器可读的表示。我们通过英语和代码翻译任务中的示例展示了它的应用，并通过针对工程师、数据科学家、用户和审计员为理论和实践提供影响。



## **11. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

Accepted by Neural Information Processing Systems (NeurIPS 2024)

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2404.13968v3) [paper-pdf](http://arxiv.org/pdf/2404.13968v3)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **12. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

LLM的普遍最优水印方案：从理论到实践 cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.02890v2) [paper-pdf](http://arxiv.org/pdf/2410.02890v2)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.

摘要: 大语言模型(LLM)提高了人类的效率，但也带来了滥用风险，水印是区分人工智能生成的内容和人类创建的文本的可靠方法。在这项工作中，我们提出了一种新的水印LLMS的理论框架。特别是，我们联合优化了水印方案和检测器以最大化检测性能，同时控制了最坏情况下的I类错误和水印文本中的失真。在我们的框架内，我们描述了普遍最小的第二类错误，显示了检测性能和失真之间的基本权衡。更重要的是，我们确定了检测器和水印方案的最佳类型。在理论分析的基础上，我们介绍了一种实用的、与模型无关的、计算高效的令牌级水印算法，该算法调用了代理模型和Gumbel-Max技巧。对Llama-13B和Mistral-8$乘以$70B的实验结果证明了该方法的有效性。此外，我们还探索了如何将稳健性融入到我们的理论框架中，这为设计未来具有更好的抗攻击能力的水印系统提供了基础。



## **13. Mind Your Questions! Towards Backdoor Attacks on Text-to-Visualization Models**

注意你的问题！对文本到可视化模型的后门攻击 cs.CR

11 pages, 4 figures

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06782v2) [paper-pdf](http://arxiv.org/pdf/2410.06782v2)

**Authors**: Shuaimin Li, Yuanfeng Song, Xuanang Chen, Anni Peng, Zhuoyue Wan, Chen Jason Zhang, Raymond Chi-Wing Wong

**Abstract**: Text-to-visualization (text-to-vis) models have become valuable tools in the era of big data, enabling users to generate data visualizations and make informed decisions through natural language queries (NLQs). Despite their widespread application, the security vulnerabilities of these models have been largely overlooked. To address this gap, we propose VisPoison, a novel framework designed to identify these vulnerabilities of current text-to-vis models systematically. VisPoison introduces two types of triggers that activate three distinct backdoor attacks, potentially leading to data exposure, misleading visualizations, or denial-of-service (DoS) incidents. The framework features both proactive and passive attack mechanisms: proactive attacks leverage rare-word triggers to access confidential data, while passive attacks, triggered unintentionally by users, exploit a first-word trigger method, causing errors or DoS events in visualizations. Through extensive experiments on both trainable and in-context learning (ICL)-based text-to-vis models, \textit{VisPoison} achieves attack success rates of over 90\%, highlighting the security problem of current text-to-vis models. Additionally, we explore two types of defense mechanisms against these attacks, but the results show that existing countermeasures are insufficient, underscoring the pressing need for more robust security solutions in text-to-vis systems.

摘要: 文本到可视化(Text-to-Vis)模型已成为大数据时代的宝贵工具，使用户能够通过自然语言查询(NLQ)生成数据可视化并做出明智的决策。尽管它们被广泛应用，但这些模型的安全漏洞在很大程度上被忽视了。为了弥补这一差距，我们提出了VisPoison，这是一个新的框架，旨在系统地识别当前文本到可视化模型的这些漏洞。VisPoison引入了两种类型的触发器，它们激活了三种不同的后门攻击，可能会导致数据泄露、误导性可视化或拒绝服务(DoS)事件。该框架同时具有主动和被动攻击机制：主动攻击利用稀有单词触发器访问机密数据，而被动攻击由用户无意触发，利用第一单词触发方法，导致可视化中的错误或DoS事件。通过对可训练文本到可视化模型和基于情景学习(ICL)的文本到可视化模型的大量实验，Texttit{VisPoison}的攻击成功率超过90%，突出了当前文本到可视化模型的安全问题。此外，我们探索了两种类型的防御机制来防御这些攻击，但结果表明现有的对策是不够的，这突显了在文本到可视化系统中迫切需要更健壮的安全解决方案。



## **14. Detecting Training Data of Large Language Models via Expectation Maximization**

通过期望最大化检测大型语言模型的训练数据 cs.CL

14 pages

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07582v1) [paper-pdf](http://arxiv.org/pdf/2410.07582v1)

**Authors**: Gyuwan Kim, Yang Li, Evangelia Spiliopoulou, Jie Ma, Miguel Ballesteros, William Yang Wang

**Abstract**: The widespread deployment of large language models (LLMs) has led to impressive advancements, yet information about their training data, a critical factor in their performance, remains undisclosed. Membership inference attacks (MIAs) aim to determine whether a specific instance was part of a target model's training data. MIAs can offer insights into LLM outputs and help detect and address concerns such as data contamination and compliance with privacy and copyright standards. However, applying MIAs to LLMs presents unique challenges due to the massive scale of pre-training data and the ambiguous nature of membership. Additionally, creating appropriate benchmarks to evaluate MIA methods is not straightforward, as training and test data distributions are often unknown. In this paper, we introduce EM-MIA, a novel MIA method for LLMs that iteratively refines membership scores and prefix scores via an expectation-maximization algorithm, leveraging the duality that the estimates of these scores can be improved by each other. Membership scores and prefix scores assess how each instance is likely to be a member and discriminative as a prefix, respectively. Our method achieves state-of-the-art results on the WikiMIA dataset. To further evaluate EM-MIA, we present OLMoMIA, a benchmark built from OLMo resources, which allows us to control the difficulty of MIA tasks with varying degrees of overlap between training and test data distributions. We believe that EM-MIA serves as a robust MIA method for LLMs and that OLMoMIA provides a valuable resource for comprehensively evaluating MIA approaches, thereby driving future research in this critical area.

摘要: 大型语言模型(LLM)的广泛应用带来了令人印象深刻的进步，但有关其训练数据的信息仍未披露，这是其性能的关键因素。成员关系推理攻击(MIA)旨在确定特定实例是否为目标模型训练数据的一部分。MIA可以提供对LLM输出的洞察，并帮助检测和解决数据污染以及遵守隐私和版权标准等问题。然而，由于大量的预培训数据和成员身份的模棱两可的性质，将MIA应用于LLMS提出了独特的挑战。此外，创建适当的基准来评估MIA方法并不简单，因为培训和测试数据分布通常是未知的。在本文中，我们介绍了EM-MIA，这是一种新的用于LLMS的MIA方法，它通过期望最大化算法迭代地精化隶属度分数和前缀分数，利用这些分数的估计可以相互提高的对偶性。成员资格分数和前缀分数分别评估每个实例作为成员的可能性和作为前缀的区别性。我们的方法在WikiMIA数据集上获得了最先进的结果。为了进一步评估EM-MIA，我们提出了OLMoMIA，一个基于OLMO资源的基准测试，它允许我们控制训练和测试数据分布之间有不同程度重叠的MIA任务的难度。我们认为，EM-MIA是一种强大的LLMS MIA方法，而OLMoMIA为全面评估MIA方法提供了宝贵的资源，从而推动了这一关键领域的未来研究。



## **15. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

简单性盛行：重新思考LLM忘记学习的负偏好优化 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07163v1) [paper-pdf](http://arxiv.org/pdf/2410.07163v1)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: In this work, we address the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences and associated model capabilities (e.g., copyrighted data or harmful content generation) while preserving essential model utilities, without the need for retraining from scratch. Despite the growing need for LLM unlearning, a principled optimization framework remains lacking. To this end, we revisit the state-of-the-art approach, negative preference optimization (NPO), and identify the issue of reference model bias, which could undermine NPO's effectiveness, particularly when unlearning forget data of varying difficulty. Given that, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We also provide deeper insights into SimNPO's advantages, supported by analysis using mixtures of Markov chains. Furthermore, we present extensive experiments validating SimNPO's superiority over existing unlearning baselines in benchmarks like TOFU and MUSE, and robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.

摘要: 在这项工作中，我们解决了大型语言模型(LLM)遗忘的问题，旨在消除不必要的数据影响和相关的模型能力(例如，受版权保护的数据或有害内容生成)，同时保留基本的模型实用程序，而不需要从头开始重新培训。尽管对LLM遗忘的需求越来越大，但仍然缺乏一个有原则的优化框架。为此，我们回顾了最先进的方法，负偏好优化(NPO)，并确定了参考模型偏差的问题，这可能会削弱NPO的有效性，特别是当遗忘遗忘数据的不同难度时。鉴于此，我们提出了一个简单而有效的遗忘优化框架，称为SimNPO，表明在消除对参考模型的依赖(通过简单偏好优化的镜头)时的“简单性”有利于遗忘。我们还提供了对SimNPO的优势的更深层次的见解，并通过使用马尔可夫链的混合分析提供了支持。此外，我们提供了大量的实验，验证了SimNPO在豆腐和缪斯等基准测试中相对于现有遗忘基线的优势，以及对重新学习攻击的健壮性。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Simple.



## **16. Universal Vulnerabilities in Large Language Models: Backdoor Attacks for In-context Learning**

大型语言模型中的普遍漏洞：上下文学习的后门攻击 cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2401.05949v6) [paper-pdf](http://arxiv.org/pdf/2401.05949v6)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Fengjun Pan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we design a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning demonstration prompts, which can make models behave in alignment with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 180B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models.

摘要: 情境学习是一种弥合预训练和微调之间差距的范式，在几个NLP任务中表现出了很高的效率，特别是在少数情况下。尽管情景学习被广泛应用，但它很容易受到恶意攻击。在这项工作中，我们提出了对此范式的安全担忧。我们的研究表明，攻击者可以通过毒化演示上下文来操纵大型语言模型的行为，而不需要对模型进行微调。具体地说，我们设计了一种新的后门攻击方法ICLAttack，用于基于上下文学习的大型语言模型。我们的方法包括两种类型的攻击：中毒演示示例和中毒演示提示，这可以使模型的行为与预定义的意图保持一致。ICLAttack不需要额外的微调来植入后门，从而保持了模型的通用性。此外，有毒的例子被正确地标记，增强了我们攻击方法的自然隐蔽性。在几个语言模型上的广泛实验结果，从1.3B到180B参数不等，证明了我们的攻击方法的有效性，例如在OPT模型上的三个数据集上的高平均攻击成功率为95.0%。



## **17. Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems**

提示感染：多代理系统内LLM到LLM提示注射 cs.MA

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07283v1) [paper-pdf](http://arxiv.org/pdf/2410.07283v1)

**Authors**: Donghyun Lee, Mo Tiwari

**Abstract**: As Large Language Models (LLMs) grow increasingly powerful, multi-agent systems are becoming more prevalent in modern AI applications. Most safety research, however, has focused on vulnerabilities in single-agent LLMs. These include prompt injection attacks, where malicious prompts embedded in external content trick the LLM into executing unintended or harmful actions, compromising the victim's application. In this paper, we reveal a more dangerous vector: LLM-to-LLM prompt injection within multi-agent systems. We introduce Prompt Infection, a novel attack where malicious prompts self-replicate across interconnected agents, behaving much like a computer virus. This attack poses severe threats, including data theft, scams, misinformation, and system-wide disruption, all while propagating silently through the system. Our extensive experiments demonstrate that multi-agent systems are highly susceptible, even when agents do not publicly share all communications. To address this, we propose LLM Tagging, a defense mechanism that, when combined with existing safeguards, significantly mitigates infection spread. This work underscores the urgent need for advanced security measures as multi-agent LLM systems become more widely adopted.

摘要: 随着大型语言模型(LLM)变得越来越强大，多智能体系统在现代人工智能应用中变得更加普遍。然而，大多数安全研究都集中在单代理LLM的漏洞上。这些攻击包括提示注入攻击，即嵌入到外部内容中的恶意提示欺骗LLM执行意外或有害的操作，从而损害受害者的应用程序。在本文中，我们揭示了一个更危险的载体：多智能体系统中的LLM到LLM快速注射。我们引入了即时感染，这是一种新型的攻击，其中恶意提示在相互连接的代理之间自我复制，行为很像计算机病毒。这种攻击构成了严重的威胁，包括数据盗窃、诈骗、错误信息和系统范围的中断，所有这些都是在系统中静默传播的。我们的大量实验表明，即使在代理不公开共享所有通信的情况下，多代理系统也是高度敏感的。为了解决这个问题，我们提出了LLM标签，这是一种防御机制，当与现有的保护措施相结合时，显著减少了感染传播。这项工作强调了随着多代理LLM系统越来越广泛地被采用，对先进安全措施的迫切需要。



## **18. Break the Visual Perception: Adversarial Attacks Targeting Encoded Visual Tokens of Large Vision-Language Models**

打破视觉感知：针对大型视觉语言模型的编码视觉标记的对抗攻击 cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06699v1) [paper-pdf](http://arxiv.org/pdf/2410.06699v1)

**Authors**: Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, Linli Xu

**Abstract**: Large vision-language models (LVLMs) integrate visual information into large language models, showcasing remarkable multi-modal conversational capabilities. However, the visual modules introduces new challenges in terms of robustness for LVLMs, as attackers can craft adversarial images that are visually clean but may mislead the model to generate incorrect answers. In general, LVLMs rely on vision encoders to transform images into visual tokens, which are crucial for the language models to perceive image contents effectively. Therefore, we are curious about one question: Can LVLMs still generate correct responses when the encoded visual tokens are attacked and disrupting the visual information? To this end, we propose a non-targeted attack method referred to as VT-Attack (Visual Tokens Attack), which constructs adversarial examples from multiple perspectives, with the goal of comprehensively disrupting feature representations and inherent relationships as well as the semantic properties of visual tokens output by image encoders. Using only access to the image encoder in the proposed attack, the generated adversarial examples exhibit transferability across diverse LVLMs utilizing the same image encoder and generality across different tasks. Extensive experiments validate the superior attack performance of the VT-Attack over baseline methods, demonstrating its effectiveness in attacking LVLMs with image encoders, which in turn can provide guidance on the robustness of LVLMs, particularly in terms of the stability of the visual feature space.

摘要: 大型视觉语言模型(LVLM)将视觉信息集成到大型语言模型中，展示了非凡的多模式对话能力。然而，视觉模块在稳健性方面为LVLMS带来了新的挑战，因为攻击者可以手工制作视觉上干净但可能误导模型生成错误答案的对抗性图像。通常，视觉编码依赖于视觉编码器将图像转换为视觉标记，这对于语言模型有效地感知图像内容是至关重要的。因此，我们好奇一个问题：当编码的视觉令牌受到攻击并扰乱视觉信息时，LVLMS还能产生正确的反应吗？为此，我们提出了一种非目标攻击方法，称为VT-Attack(视觉标记攻击)，它从多个角度构造对抗性实例，目的是综合破坏图像编码者输出的视觉标记的特征表示和内在关系以及语义属性。在所提出的攻击中，仅使用对图像编码器的访问，生成的敌意示例表现出在使用相同图像编码器的不同LVLM之间的可转移性和跨不同任务的通用性。大量的实验验证了VT攻击相对于基线方法的优越攻击性能，证明了其在利用图像编码器攻击LVLM方面的有效性，进而可以为LVLMS的稳健性，特别是视觉特征空间的稳定性提供指导。



## **19. FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services**

FELLAS：以LLM作为外部服务增强联合顺序推荐 cs.IR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.04927v2) [paper-pdf](http://arxiv.org/pdf/2410.04927v2)

**Authors**: Wei Yuan, Chaoqun Yang, Guanhua Ye, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated sequential recommendation (FedSeqRec) has gained growing attention due to its ability to protect user privacy. Unfortunately, the performance of FedSeqRec is still unsatisfactory because the models used in FedSeqRec have to be lightweight to accommodate communication bandwidth and clients' on-device computational resource constraints. Recently, large language models (LLMs) have exhibited strong transferable and generalized language understanding abilities and therefore, in the NLP area, many downstream tasks now utilize LLMs as a service to achieve superior performance without constructing complex models. Inspired by this successful practice, we propose a generic FedSeqRec framework, FELLAS, which aims to enhance FedSeqRec by utilizing LLMs as an external service. Specifically, FELLAS employs an LLM server to provide both item-level and sequence-level representation assistance. The item-level representation service is queried by the central server to enrich the original ID-based item embedding with textual information, while the sequence-level representation service is accessed by each client. However, invoking the sequence-level representation service requires clients to send sequences to the external LLM server. To safeguard privacy, we implement dx-privacy satisfied sequence perturbation, which protects clients' sensitive data with guarantees. Additionally, a contrastive learning-based method is designed to transfer knowledge from the noisy sequence representation to clients' sequential recommendation models. Furthermore, to empirically validate the privacy protection capability of FELLAS, we propose two interacted item inference attacks. Extensive experiments conducted on three datasets with two widely used sequential recommendation models demonstrate the effectiveness and privacy-preserving capability of FELLAS.

摘要: 联邦顺序推荐(FedSeqRec)由于其保护用户隐私的能力而受到越来越多的关注。遗憾的是，FedSeqRec的性能仍然不能令人满意，因为FedSeqRec中使用的模型必须是轻量级的，以适应通信带宽和客户端在设备上的计算资源限制。近年来，大语言模型表现出很强的可迁移和泛化语言理解能力，因此，在自然语言处理领域，许多下游任务现在将大语言模型作为一种服务来获得优越的性能，而不需要构建复杂的模型。受这一成功实践的启发，我们提出了一个通用的FedSeqRec框架Fellas，旨在通过利用LLMS作为外部服务来增强FedSeqRec。具体地说，FELLAS使用LLM服务器来提供物品级和序列级的表示帮助。项级表示服务由中央服务器查询，以丰富嵌入文本信息的原始基于ID的项，而序列级表示服务由每个客户端访问。但是，调用序列级别表示服务需要客户端将序列发送到外部LLM服务器。为了保护隐私，我们实现了满足DX隐私的序列扰动，用保证来保护客户的敏感数据。此外，设计了一种基于对比学习的方法来将知识从噪声序列表示转移到客户的序贯推荐模型。此外，为了经验性地验证Fellas的隐私保护能力，我们提出了两种交互的项目推理攻击。在三个数据集和两个广泛使用的序列推荐模型上进行了大量的实验，证明了Fellas的有效性和隐私保护能力。



## **20. Signal Watermark on Large Language Models**

大型语言模型上的信号水印 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06545v1) [paper-pdf](http://arxiv.org/pdf/2410.06545v1)

**Authors**: Zhenyu Xu, Victor S. Sheng

**Abstract**: As Large Language Models (LLMs) become increasingly sophisticated, they raise significant security concerns, including the creation of fake news and academic misuse. Most detectors for identifying model-generated text are limited by their reliance on variance in perplexity and burstiness, and they require substantial computational resources. In this paper, we proposed a watermarking method embedding a specific watermark into the text during its generation by LLMs, based on a pre-defined signal pattern. This technique not only ensures the watermark's invisibility to humans but also maintains the quality and grammatical integrity of model-generated text. We utilize LLMs and Fast Fourier Transform (FFT) for token probability computation and detection of the signal watermark. The unique application of signal processing principles within the realm of text generation by LLMs allows for subtle yet effective embedding of watermarks, which do not compromise the quality or coherence of the generated text. Our method has been empirically validated across multiple LLMs, consistently maintaining high detection accuracy, even with variations in temperature settings during text generation. In the experiment of distinguishing between human-written and watermarked text, our method achieved an AUROC score of 0.97, significantly outperforming existing methods like GPTZero, which scored 0.64. The watermark's resilience to various attacking scenarios further confirms its robustness, addressing significant challenges in model-generated text authentication.

摘要: 随着大型语言模型(LLM)变得越来越复杂，它们引发了重大的安全问题，包括制造假新闻和学术滥用。大多数用于识别模型生成的文本的检测器都受到其对困惑和突发性变化的依赖的限制，并且它们需要大量的计算资源。本文提出了一种基于预定义的信号模式，在LLMS生成文本的过程中嵌入特定水印的水印方法。该技术不仅保证了水印对人类的不可见性，而且保持了模型生成文本的质量和语法完整性。我们利用LLMS和快速傅立叶变换(FFT)进行令牌概率计算和信号水印检测。LLMS在文本生成领域独特地应用信号处理原理，允许微妙而有效地嵌入水印，这不会损害生成的文本的质量或连贯性。我们的方法已经在多个LLM上进行了经验验证，即使在文本生成期间温度设置变化的情况下，也始终保持高检测精度。在区分人写文本和带水印文本的实验中，我们的方法达到了0.97的AUROC分数，远远超过了GPTZero等现有的方法，GPTZero的分数为0.64。水印对各种攻击场景的弹性进一步证实了它的健壮性，解决了模型生成的文本身份验证中的重大挑战。



## **21. WAPITI: A Watermark for Finetuned Open-Source LLMs**

WAPITI：Finetuned开源LLM的水印 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06467v1) [paper-pdf](http://arxiv.org/pdf/2410.06467v1)

**Authors**: Lingjie Chen, Ruizhong Qiu, Siyu Yuan, Zhining Liu, Tianxin Wei, Hyunsik Yoo, Zhichen Zeng, Deqing Yang, Hanghang Tong

**Abstract**: Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.

摘要: 大语言模型(LLMS)水印生成在文本中嵌入了一种不可察觉的统计模式，使其在算法上是可检测的。水印是一种很有前途的方法，可以解决LLMS的潜在危害和偏见，因为它能够跟踪、问责和检测被篡改的内容，有助于减轻意外后果。然而，对于开源模型，水印面临着两大挑战：(I)与微调模型不兼容，(Ii)易受微调攻击。在这项工作中，我们提出了Wapiti，一种新的方法，通过参数积分将水印从基本模型转移到微调模型。就我们所知，我们建议为保持其微调能力的开放源码LLM提供第一个水印。此外，我们的方法提供了针对微调攻击的有效防御。我们在不同的模型架构和水印策略上测试了我们的方法。实验结果表明，该方法能够成功地嵌入水印，并且与微调模型具有很好的兼容性。此外，我们还深入分析了参数编辑如何影响最终模型的水印强度和整体性能。



## **22. Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders**

幻觉人工智能劫持攻击：大型语言模型和恶意代码推荐 cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06462v1) [paper-pdf](http://arxiv.org/pdf/2410.06462v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: The research builds and evaluates the adversarial potential to introduce copied code or hallucinated AI recommendations for malicious code in popular code repositories. While foundational large language models (LLMs) from OpenAI, Google, and Anthropic guard against both harmful behaviors and toxic strings, previous work on math solutions that embed harmful prompts demonstrate that the guardrails may differ between expert contexts. These loopholes would appear in mixture of expert's models when the context of the question changes and may offer fewer malicious training examples to filter toxic comments or recommended offensive actions. The present work demonstrates that foundational models may refuse to propose destructive actions correctly when prompted overtly but may unfortunately drop their guard when presented with a sudden change of context, like solving a computer programming challenge. We show empirical examples with trojan-hosting repositories like GitHub, NPM, NuGet, and popular content delivery networks (CDN) like jsDelivr which amplify the attack surface. In the LLM's directives to be helpful, example recommendations propose application programming interface (API) endpoints which a determined domain-squatter could acquire and setup attack mobile infrastructure that triggers from the naively copied code. We compare this attack to previous work on context-shifting and contrast the attack surface as a novel version of "living off the land" attacks in the malware literature. In the latter case, foundational language models can hijack otherwise innocent user prompts to recommend actions that violate their owners' safety policies when posed directly without the accompanying coding support request.

摘要: 这项研究构建并评估了在流行的代码库中引入复制代码或幻觉AI建议的恶意代码的敌意潜力。虽然OpenAI、谷歌和人类的基础大型语言模型(LLM)可以防范有害行为和有毒字符串，但之前关于嵌入有害提示的数学解决方案的工作表明，护栏可能会因专家上下文而异。当问题的上下文发生变化时，这些漏洞将出现在专家模型的混合中，并且可能提供较少的恶意训练示例来过滤有毒评论或建议的攻击性操作。目前的工作表明，基础模型可能会在公开提示时拒绝正确地提出破坏性行动，但不幸的是，当环境突然改变时，可能会放松警惕，比如解决计算机编程挑战。我们使用GitHub、NPM、NuGet等木马托管库和jsDelivr等流行的内容交付网络(CDN)展示了放大攻击面的经验示例。在LLM的有用指令中，示例建议提出了应用程序编程接口(API)端点，确定的域抢占者可以获取这些端点，并建立从简单复制的代码触发的攻击移动基础设施。我们将这一攻击与之前关于上下文转换的工作进行了比较，并将攻击面作为恶意软件文献中的一个新版本的“赖以生存”的攻击进行了对比。在后一种情况下，基础语言模型可以劫持其他无辜的用户提示，在没有附带的编码支持请求的情况下直接提出违反其所有者安全政策的行为。



## **23. Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning**

评估和保障基于检索的上下文学习的对抗鲁棒性 cs.CL

COLM 2024, 31 pages, 6 figures

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.15984v4) [paper-pdf](http://arxiv.org/pdf/2405.15984v4)

**Authors**: Simon Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **24. Training-free LLM-generated Text Detection by Mining Token Probability Sequences**

通过挖掘令牌概率序列进行免训练LLM生成的文本检测 cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.06072v1) [paper-pdf](http://arxiv.org/pdf/2410.06072v1)

**Authors**: Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, Fei Wu

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in generating high-quality texts across diverse domains. However, the potential misuse of LLMs has raised significant concerns, underscoring the urgent need for reliable detection of LLM-generated texts. Conventional training-based detectors often struggle with generalization, particularly in cross-domain and cross-model scenarios. In contrast, training-free methods, which focus on inherent discrepancies through carefully designed statistical features, offer improved generalization and interpretability. Despite this, existing training-free detection methods typically rely on global text sequence statistics, neglecting the modeling of local discriminative features, thereby limiting their detection efficacy. In this work, we introduce a novel training-free detector, termed \textbf{Lastde} that synergizes local and global statistics for enhanced detection. For the first time, we introduce time series analysis to LLM-generated text detection, capturing the temporal dynamics of token probability sequences. By integrating these local statistics with global ones, our detector reveals significant disparities between human and LLM-generated texts. We also propose an efficient alternative, \textbf{Lastde++} to enable real-time detection. Extensive experiments on six datasets involving cross-domain, cross-model, and cross-lingual detection scenarios, under both white-box and black-box settings, demonstrated that our method consistently achieves state-of-the-art performance. Furthermore, our approach exhibits greater robustness against paraphrasing attacks compared to existing baseline methods.

摘要: 大型语言模型(LLM)在跨不同领域生成高质量文本方面表现出了非凡的能力。然而，LLMS的潜在滥用引起了重大关切，突显了迫切需要可靠地检测LLMS生成的文本。传统的基于训练的检测器经常难以泛化，特别是在跨域和跨模型的场景中。相比之下，无需训练的方法通过精心设计的统计特征侧重于内在差异，提供了更好的概括性和可解释性。尽管如此，现有的免训练检测方法通常依赖于全局文本序列统计，忽略了对局部区分特征的建模，从而限制了它们的检测效率。在这项工作中，我们引入了一种新的无需训练的检测器，称为Textbf{Lastde}，它协同局部和全局统计来增强检测。首次将时间序列分析引入到LLM生成的文本检测中，捕捉了令牌概率序列的时间动态。通过将这些局部统计数据与全球统计数据相结合，我们的检测器发现了人类和LLM生成的文本之间的显著差异。我们还提出了一种有效的替代方案，\extbf{Lastde++}来实现实时检测。在白盒和黑盒环境下，在六个涉及跨域、跨模型和跨语言检测场景的数据集上的大量实验表明，我们的方法始终达到了最先进的性能。此外，与现有的基线方法相比，我们的方法对意译攻击表现出更好的稳健性。



## **25. Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs**

针对LLM的有效且规避的模糊测试驱动越狱攻击 cs.CR

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2409.14866v2) [paper-pdf](http://arxiv.org/pdf/2409.14866v2)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates, our method starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks.   We evaluated our method on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, our method achieves attack success rates of over 90%,80% and 74%, respectively, exceeding existing baselines by more than 60%. Additionally, our method can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, our method can achieve over 78% attack success rate even with 100 tokens. Moreover, our method demonstrates transferability and is robust to state-of-the-art defenses. We will open-source our codes upon publication.

摘要: 大型语言模型(LLM)在各种任务中表现出色，但仍然容易受到越狱攻击，在越狱攻击中，攻击者创建越狱提示来误导模型生成有害或攻击性内容。当前的越狱方法要么严重依赖于人工制作的模板，这对可伸缩性和适应性构成了挑战，要么难以生成语义连贯的提示，使它们很容易被检测到。针对现有方法存在提示过长、查询代价较高的问题，提出了一种新的越狱攻击框架，该框架采用了一系列定制的黑盒模糊测试方法，是一种自动化的黑盒越狱攻击框架。我们的方法不依赖于手动创建的模板，而是从一个空的种子池开始，不需要搜索任何相关的越狱模板。我们还开发了三种新的问题相关突变策略，使用LLM助手来生成提示，这些提示在保持语义连贯的同时显著缩短了提示的长度。此外，我们实现了一个两级判断模块来准确地检测真正的成功越狱。我们在7个有代表性的LLM上对我们的方法进行了评估，并将其与5种最先进的越狱攻击策略进行了比较。对于专有的LLMAPI，如GPT-3.5 Turbo、GPT-4和Gemini-Pro，我们的方法分别实现了90%、80%和74%以上的攻击成功率，比现有基线高出60%以上。此外，我们的方法可以保持较高的语义一致性，同时显著减少越狱提示的长度。在攻击GPT-4时，即使有100个令牌，我们的方法也可以达到78%以上的攻击成功率。此外，我们的方法证明了可转移性，并对最先进的防御措施具有健壮性。我们将在发布后将我们的代码开源。



## **26. You Know What I'm Saying: Jailbreak Attack via Implicit Reference**

你知道我在说什么：通过隐性引用进行越狱攻击 cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.03857v2) [paper-pdf](http://arxiv.org/pdf/2410.03857v2)

**Authors**: Tianyu Wu, Lingrui Mei, Ruibin Yuan, Lujun Li, Wei Xue, Yike Guo

**Abstract**: While recent advancements in large language model (LLM) alignment have enabled the effective identification of malicious objectives involving scene nesting and keyword rewriting, our study reveals that these methods remain inadequate at detecting malicious objectives expressed through context within nested harmless objectives. This study identifies a previously overlooked vulnerability, which we term Attack via Implicit Reference (AIR). AIR decomposes a malicious objective into permissible objectives and links them through implicit references within the context. This method employs multiple related harmless objectives to generate malicious content without triggering refusal responses, thereby effectively bypassing existing detection techniques.Our experiments demonstrate AIR's effectiveness across state-of-the-art LLMs, achieving an attack success rate (ASR) exceeding 90% on most models, including GPT-4o, Claude-3.5-Sonnet, and Qwen-2-72B. Notably, we observe an inverse scaling phenomenon, where larger models are more vulnerable to this attack method. These findings underscore the urgent need for defense mechanisms capable of understanding and preventing contextual attacks. Furthermore, we introduce a cross-model attack strategy that leverages less secure models to generate malicious contexts, thereby further increasing the ASR when targeting other models.Our code and jailbreak artifacts can be found at https://github.com/Lucas-TY/llm_Implicit_reference.

摘要: 虽然最近在大语言模型(LLM)对齐方面的进展使得能够有效地识别涉及场景嵌套和关键字重写的恶意目标，但我们的研究表明，这些方法在检测嵌套的无害目标中通过上下文表达的恶意目标方面仍然不足。这项研究发现了一个以前被忽视的漏洞，我们将其称为隐式引用攻击(AIR)。AIR将恶意目标分解为允许的目标，并通过上下文中的隐式引用将它们链接起来。该方法利用多个相关的无害目标在不触发拒绝响应的情况下生成恶意内容，从而有效地绕过了现有的检测技术。我们的实验证明了AIR在最先进的LLM上的有效性，在包括GPT-40、Claude-3.5-Sonnet和Qwen-2-72B在内的大多数型号上实现了超过90%的攻击成功率(ASR)。值得注意的是，我们观察到了反向缩放现象，其中较大的模型更容易受到这种攻击方法的攻击。这些发现突显了迫切需要能够理解和防止上下文攻击的防御机制。此外，我们引入了一种跨模型攻击策略，该策略利用安全性较低的模型来生成恶意上下文，从而进一步提高了针对其他模型的ASR。我们的代码和越狱人工产物可以在https://github.com/Lucas-TY/llm_Implicit_reference.找到



## **27. Partially Recentralization Softmax Loss for Vision-Language Models Robustness**

部分再集中化Softmax因视觉语言模型鲁棒性而丧失 cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2402.03627v2) [paper-pdf](http://arxiv.org/pdf/2402.03627v2)

**Authors**: Hao Wang, Jinzhe Jiang, Xin Zhang, Chen Li

**Abstract**: As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted

摘要: 随着大型语言模型在自然语言处理任务(NLP)方面的突破，多通道技术变得非常流行。然而，已有研究表明，多模式NLP很容易受到对抗性攻击，模型的输出可能会因输入的扰动而发生显著变化。虽然在计算机视觉和NLP模型中已经提出了几种防御技术，但模型的多通道稳健性还没有得到充分的研究。在本文中，我们研究了通过限制Top K Softmax输出来修改预先训练的多模式模型的损失函数所提供的对抗鲁棒性。在评估和评分的基础上，我们的实验表明，经过微调后，预先训练的模型对攻击的健壮性可以显著提高，对抗流行攻击。这类损失函数的输出分集、泛化以及稳健性与性能的权衡等问题还有待进一步研究。我们的代码将在这篇论文被接受后可用



## **28. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

ATM：对抗性调整多代理系统打造强大的检索增强生成器 cs.CL

18 pages

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.18111v3) [paper-pdf](http://arxiv.org/pdf/2405.18111v3)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, since today's Internet is flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with an Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent through adversarially tuning the agents for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to the state-of-the-art baselines.

摘要: 事实证明，大型语言模型(LLM)在缓解面对知识密集型问题时的幻觉方面，从检索增强生成(RAG)中受益匪浅。RAG采用信息检索技术，从与语义相关的文档中注入外部知识作为输入上下文。然而，由于当今的互联网充斥着大量噪声和捏造的内容，RAG系统不可避免地容易受到这些噪声的影响，并容易做出错误的响应。为此，我们提出了用对抗性调谐多智能体系统(ATM)来优化检索增强生成器。ATM通过对代理进行多次恶意调整，在辅助攻击者代理的帮助下，引导Generator具有用于问题回答的有用文档的健壮视角。经过几轮多代理迭代调整后，Generator最终可以更好地区分有用的文档和捏造的文档。实验结果验证了ATM的有效性，并且我们还观察到，与最先进的基线相比，该生成器可以获得更好的性能。



## **29. Aligning LLMs to Be Robust Against Prompt Injection**

调整LLM以应对即时注入的稳健性 cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications. Alignment training makes LLMs robust against even the strongest  prompt injection attacks

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05451v1) [paper-pdf](http://arxiv.org/pdf/2410.05451v1)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be carefully crafted and injected into external data sources to override the user's intended instruction and instead execute a malicious instruction. Prompt injection attacks constitute a major threat to LLM security, making the design and implementation of practical countermeasures of paramount importance. To this end, we show that alignment can be a powerful tool to make LLMs more robust against prompt injection. Our method -- SecAlign -- first builds an alignment dataset by simulating prompt injection attacks and constructing pairs of desirable and undesirable responses. Then, we apply existing alignment techniques to fine-tune the LLM to be robust against these simulated attacks. Our experiments show that SecAlign robustifies the LLM substantially with a negligible hurt on model utility. Moreover, SecAlign's protection generalizes to strong attacks unseen in training. Specifically, the success rate of state-of-the-art GCG-based prompt injections drops from 56% to 2% in Mistral-7B after our alignment process. Our code is released at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型(LLM)在现代软件系统中正变得越来越普遍，它们在用户和互联网之间进行接口，以帮助完成需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，如用户文档、Web检索、API调用结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。恶意提示可以精心编制并注入外部数据源，以覆盖用户的预期指令，而不是执行恶意指令。快速注入攻击是对LLM安全的主要威胁，因此设计和实施实用的对策至关重要。为此，我们表明对齐可以是一个强大的工具，以使LLM更健壮地抵御快速注入。我们的方法--SecAlign--首先通过模拟即时注入攻击并构建期望和不期望的响应对来构建比对数据集。然后，我们应用现有的对准技术来微调LLM，使其对这些模拟攻击具有健壮性。我们的实验表明，SecAlign在很大程度上增强了LLM的健壮性，而对模型效用的损害可以忽略不计。此外，SecAlign的保护可以概括为在训练中看不到的强大攻击。具体地说，在我们的对准过程之后，米斯特拉尔-7B最先进的基于GCG的快速注射的成功率从56%下降到2%。我们的代码在https://github.com/facebookresearch/SecAlign上发布



## **30. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models are Covert Channels**

$$\mathBF{L ' 2\csot M = C ' 2}$$大型语言模型是秘密渠道 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2405.15652v2) [paper-pdf](http://arxiv.org/pdf/2405.15652v2)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity recently. LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship-resistant communication? In this paper, we explore open-source LLM-based covert channels. We empirically measure the security vs. capacity of an open-source LLM model (Llama-7B) to assess its performance as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, we also show that the chance for an adversary to detect covert communication is low. To ensure our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.

摘要: 大型语言模型（LLM）最近受到广泛欢迎。LLM容易受到各种攻击，但也可以提高不同系统的安全性。然而，除了支持更安全的系统之外，开源LLM作为covertext分发版的表现如何，例如促进抗审查的沟通？在本文中，我们探索基于开源LLM的隐蔽渠道。我们根据经验测量开源LLM模型（Llama-7 B）的安全性与容量，以评估其作为隐蔽渠道的性能。尽管我们的结果表明此类通道不太可能实现高的实际比特率，但我们也表明对手检测秘密通信的机会很低。为了确保我们的结果可以以最少的努力作为一般参考，我们采用了概念上简单且简洁的方案，并且仅假设公共模型。



## **31. Representation noising effectively prevents harmful fine-tuning on LLMs**

表示噪音有效防止对LLM的有害微调 cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2405.14577v2) [paper-pdf](http://arxiv.org/pdf/2405.14577v2)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that is effective even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the effectiveness of our defence lies in its "depth": the degree to which information about harmful representations is removed across all layers of the LLM.

摘要: 发布开源的大型语言模型(LLM)存在双重用途的风险，因为不好的参与者很容易出于有害目的微调这些模型。即使没有公开的权重释放，权重盗窃和微调API也会使封闭的模型容易受到有害的微调攻击(HFA)。虽然防止越狱和改善安全护栏等安全措施很重要，但通过微调很容易逆转这些措施。在这项工作中，我们提出了表示噪声(RepNoise)，这是一种即使攻击者可以访问权重也有效的防御机制。RepNoise的工作原理是删除有关有害表示的信息，以便在微调期间很难恢复它们。重要的是，我们的防御还能够概括在防御过程中未曾见过的伤害的不同子集，只要它们来自相同分布的攻击集。我们的方法不会降低LLMS的整体性能，并保留了对模型进行无害任务训练的能力。我们提供的经验证据表明，我们辩护的有效性在于它的“深度”：在LLM的所有层中，有关有害陈述的信息被移除的程度。



## **32. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2312.03853v5) [paper-pdf](http://arxiv.org/pdf/2312.03853v5)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbots. Safety mechanisms are implemented to prevent improper responses from these chatbots. In this work, we bypass these measures for ChatGPT and Gemini by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information in both ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, showing that both chatbots are vulnerable to this attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.

摘要: 最近，我们看到大型语言模型(LLM)的使用有所增加，特别是在聊天机器人等应用程序中。安全机制的实施是为了防止这些聊天机器人做出不适当的反应。在这项工作中，我们绕过了ChatGPT和Gemini的这些措施，让他们模仿具有与诚实的助手不一致的人格特征的复杂人物角色。首先，我们为这些角色创建详细的传记，然后在与相同聊天机器人的新会话中使用这些传记。然后，我们的对话遵循角色扮演的风格，以引发被禁止的回应。使用人物角色，我们显示提供了禁止的响应，使得在ChatGPT和Gemini中获取未经授权的、非法的或有害的信息成为可能。我们还介绍了几种激活这种敌对角色的方法，表明这两个聊天机器人都容易受到这种攻击。在相同的原则下，我们引入了两个防御措施，推动该模型解释可信任的个性，并使其对此类攻击更加健壮。



## **33. Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models**

重建您之前的对话！全面调查GPT模型对话中的隐私泄露风险 cs.CR

Accepted in EMNLP 2024. 14 pages, 10 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2402.02987v2) [paper-pdf](http://arxiv.org/pdf/2402.02987v2)

**Authors**: Junjie Chu, Zeyang Sha, Michael Backes, Yang Zhang

**Abstract**: Significant advancements have recently been made in large language models represented by GPT models. Users frequently have multi-round private conversations with cloud-hosted GPT models for task optimization. Yet, this operational paradigm introduces additional attack surfaces, particularly in custom GPTs and hijacked chat sessions. In this paper, we introduce a straightforward yet potent Conversation Reconstruction Attack. This attack targets the contents of previous conversations between GPT models and benign users, i.e., the benign users' input contents during their interaction with GPT models. The adversary could induce GPT models to leak such contents by querying them with designed malicious prompts. Our comprehensive examination of privacy risks during the interactions with GPT models under this attack reveals GPT-4's considerable resilience. We present two advanced attacks targeting improved reconstruction of past conversations, demonstrating significant privacy leakage across all models under these advanced techniques. Evaluating various defense mechanisms, we find them ineffective against these attacks. Our findings highlight the ease with which privacy can be compromised in interactions with GPT models, urging the community to safeguard against potential abuses of these models' capabilities.

摘要: 最近，以GPT模型为代表的大型语言模型取得了重大进展。用户经常与云托管的GPT模型进行多轮私下对话，以实现任务优化。然而，这种操作模式引入了额外的攻击面，特别是在定制GPT和被劫持的聊天会话中。在本文中，我们介绍了一种简单而有效的会话重建攻击。该攻击的目标是GPT模型与良性用户之间先前对话的内容，即良性用户在与GPT模型交互时的输入内容。攻击者可以通过设计恶意提示来查询GPT模型，从而诱导GPT模型泄露此类内容。我们对在这次攻击下与GPT模型交互过程中的隐私风险进行了全面的检查，发现GPT-4的S具有相当强的韧性。我们提出了两个高级攻击，目标是改进过去对话的重建，表明在这些高级技术下，所有模型都存在显著的隐私泄露。评估各种防御机制，我们发现它们对这些攻击无效。我们的发现突出了隐私在与GPT模型的交互中很容易受到损害，敦促社区防范这些模型的功能可能被滥用。



## **34. AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models**

AnyAttack：面向视觉语言模型的大规模自我监督生成有针对性的对抗示例 cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05346v1) [paper-pdf](http://arxiv.org/pdf/2410.05346v1)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack, a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack. To address the limitation of existing methods that require label supervision, we introduce a contrastive loss that trains a generator on a large-scale unlabeled image dataset, LAION-400M dataset, for generating targeted adversarial noise. This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs. Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack. Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google's Gemini, Claude's Sonnet, and Microsoft's Copilot. These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.

摘要: 由于其多通道能力，视觉语言模型(VLM)在现实世界场景中发现了许多有影响力的应用。然而，最近的研究表明，VLM很容易受到基于图像的敌意攻击，特别是针对操纵模型以生成对手指定的有害内容的对抗性图像。当前的攻击方法依赖于预定义的目标标签来创建有针对性的对抗性攻击，这限制了它们在大规模健壮性评估中的可扩展性和适用性。在本文中，我们提出了AnyAttack，这是一个自监督框架，可以在没有标签监督的情况下为VLMS生成有针对性的敌意图像，允许任何图像作为攻击的目标。为了解决现有方法需要标签监督的局限性，我们引入了一种对比损失，它在大规模的未标记图像数据集LAION-400M数据集上训练生成器，以生成目标对抗性噪声。这种大规模的预培训使我们的方法在广泛的VLM中具有强大的可移植性。在三个多模式任务(图像-文本检索、多模式分类和图像字幕)上对五个主流开源VLMS(CLIP、BLIP、BLIP2、InstructBLIP和MiniGPT-4)进行了广泛的实验，证明了该攻击的有效性。此外，我们成功地将AnyAttack转移到多个商业VLM上，包括谷歌的Gemini、Claude的十四行诗和微软的Copilot。这些结果揭示了极小武器系统面临的前所未有的风险，突显了采取有效对策的必要性。



## **35. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

越狱解药：通过大型语言模型中的稀疏表示调整来实现安全与效用平衡 cs.CR

10 pages, 5 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.02298v2) [paper-pdf](http://arxiv.org/pdf/2410.02298v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.

摘要: 随着大型语言模型(LLM)成为各种应用程序不可或缺的一部分，确保它们的安全性和实用性是至关重要的。越狱攻击操纵LLM生成有害内容，对这种平衡构成了重大挑战。现有的防御措施，如即时工程和安全微调，通常会引入计算开销，增加推理延迟，并且缺乏运行时灵活性。此外，过于严格的安全措施可能会导致良性查询被拒绝，从而降低模型的实用性。在本文中，我们介绍了JailBreak解毒剂，这是一种通过在推理过程中操纵模型内部状态的稀疏子集来实时调整LLM安全偏好的方法。通过沿不同强度的安全方向移动模型的隐藏表示，我们在不增加令牌开销或推理延迟的情况下实现了对安全-效用平衡的灵活控制。我们的分析表明，LLMS中与安全相关的信息是稀疏分布的；调整大约5%的内部状态与修改整个状态一样有效。在9个LLM(参数从20亿到720亿)上进行了大量的实验，对10种越狱攻击方法进行了评估，并与6种防御策略进行了比较，验证了该方法的有效性和高效性。通过在推理过程中直接操纵内部状态，越狱解毒剂提供了一个轻量级、可扩展的解决方案，在增强LLM安全性的同时保留了实用性，为广泛部署的AI系统中的实时安全机制打开了新的可能性。



## **36. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

CleanGen：缓解大型语言模型中生成任务的后门攻击 cs.AI

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2406.12257v2) [paper-pdf](http://arxiv.org/pdf/2406.12257v2)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CLEANGEN, to mitigate backdoor attacks for generation tasks in LLMs. CLEANGEN is a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CLEANGEN is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CLEANGEN against five SOTA backdoor attacks. Our results show that CLEANGEN achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CLEANGEN maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.

摘要: 大型语言模型(LLM)在生成任务中的出色性能使实践者能够利用公开可用的模型来支持定制应用程序，如聊天机器人和虚拟助手。然而，用于训练或微调这些LLM的数据往往是秘密的，这使得攻击者能够危害数据并向模型注入后门。本文提出了一种新的推理时间防御机制CLEANGEN，用于缓解LLMS中针对生成任务的后门攻击。CLEANGEN是一种轻量级且有效的解码策略，与最先进的(SOTA)LLMS兼容。CLEANGEN背后的洞见是，与其他LLM相比，后置LLM为代表攻击者所需内容的令牌分配了显著更高的概率。令牌概率中的这些差异使CLEANGEN能够识别攻击者偏爱的可疑令牌，并将其替换为由另一个未被同一攻击者破解的LLM生成的令牌，从而避免生成攻击者所需的内容。我们评估了Cleangen对五次Sota后门攻击的效果。我们的结果表明，对于所有五个后门攻击，CLEANGEN实现了比五个SOTA基线防御更低的攻击成功率(ASR)。此外，部署CLEANGEN的LLMS在以最小的额外计算开销服务于良性用户查询时，在其响应中保持了帮助。



## **37. Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm**

遗忘：消除参数高效微调范式中的任务不可知后门 cs.CL

Under Review

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2409.14119v3) [paper-pdf](http://arxiv.org/pdf/2409.14119v3)

**Authors**: Jaehan Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a key training strategy for large language models. However, its reliance on fewer trainable parameters poses security risks, such as task-agnostic backdoors. Despite their severe impact on a wide range of tasks, there is no practical defense solution available that effectively counters task-agnostic backdoors within the context of PEFT. In this study, we introduce Obliviate, a PEFT-integrable backdoor defense. We develop two techniques aimed at amplifying benign neurons within PEFT layers and penalizing the influence of trigger tokens. Our evaluations across three major PEFT architectures show that our method can significantly reduce the attack success rate of the state-of-the-art task-agnostic backdoors (83.6%$\downarrow$). Furthermore, our method exhibits robust defense capabilities against both task-specific backdoors and adaptive attacks. Source code will be obtained at https://github.com/obliviateARR/Obliviate.

摘要: 参数高效微调（PEFT）已成为大型语言模型的关键训练策略。然而，它对较少可训练参数的依赖会带来安全风险，例如任务不可知的后门。尽管它们对广泛的任务产生了严重影响，但没有实用的防御解决方案可以有效地对抗PEFT背景下的任务不可知后门。在这项研究中，我们引入Obliviate，一种PEFT可集成的后门防御。我们开发了两种技术，旨在放大PEFT层内的良性神经元并惩罚触发代币的影响。我们对三种主要PEFT架构的评估表明，我们的方法可以显着降低最先进的任务不可知后门（83.6%$\down arrow $）的攻击成功率。此外，我们的方法对特定任务的后门和自适应攻击都表现出强大的防御能力。源代码可在https://github.com/obliviateARR/Obliviate上获取。



## **38. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

AppPoet：通过多视图提示工程进行基于大语言模型的Android恶意软件检测 cs.CR

**SubmitDate**: 2024-10-06    [abs](http://arxiv.org/abs/2404.18816v2) [paper-pdf](http://arxiv.org/pdf/2404.18816v2)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous learning-based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing learning-based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Then, using our carefully crafted multi-view prompt templates, it guides the LLM to generate function descriptions and behavioral summaries for each view, enabling deep semantic analysis of the views. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the human-readable diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline methods. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.

摘要: 由于Android应用种类繁多，功能多样，行为语义错综复杂，攻击者可以采取各种策略，将真实的攻击意图隐藏在合法的功能中。然而，许多基于学习的方法在挖掘行为语义信息方面存在局限性，从而阻碍了Android恶意软件检测的准确性和效率。此外，现有的基于学习的方法大多解释性较弱，不能为研究人员提供有效的、可读性强的检测报告。受大语言模型在自然语言理解方面的成功启发，我们提出了一种基于大语言模型的Android恶意软件检测系统AppPoet。首先，AppPoet使用静态的方法来全面收集应用程序的特征，并制定各种观察视图。然后，使用我们精心设计的多视图提示模板，它指导LLM为每个视图生成功能描述和行为摘要，从而实现对视图的深入语义分析。最后，通过深度神经网络(DNN)分类器对多视图信息进行协同融合，高效准确地检测出恶意软件，并生成人类可读的诊断报告。实验结果表明，该方法的检测正确率为97.15%，F1评分为97.21%，优于基线方法。此外，案例研究还评估了我们生成的诊断报告的有效性。



## **39. Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks**

功能同伦：通过LLM越狱攻击的连续参数平滑离散优化 cs.LG

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04234v1) [paper-pdf](http://arxiv.org/pdf/2410.04234v1)

**Authors**: Zi Wang, Divyam Anshumaan, Ashish Hooda, Yudong Chen, Somesh Jha

**Abstract**: Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.

摘要: 优化方法广泛应用于深度学习中，以识别和减轻不希望的模型响应。虽然基于梯度的技术已被证明对图像模型有效，但它们在语言模型中的应用受到输入空间的离散性的阻碍。这项研究引入了一种新型优化方法，称为\{函数同伦}方法，它利用了模型训练和输入生成之间的函数二元性。通过构建一系列容易到难的优化问题，我们使用从已建立的同伦方法推导出的原则迭代解决这些问题。我们将这种方法应用于大型语言模型（LLM）的越狱攻击合成，在规避已建立的安全开源模型（例如Llama-2和Llama-3）方面，与现有方法相比，成功率提高了20 -30美元。



## **40. Adversarial Suffixes May Be Features Too!**

敌对后缀也可能是功能！ cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.00451v2) [paper-pdf](http://arxiv.org/pdf/2410.00451v2)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including those triggered by adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets, i.e., even in the absence of harmful content. This highlights the critical risk posed by dominating benign features in the training data and calls for further research to reinforce LLM safety alignment. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.

摘要: 尽管在安全匹配方面正在进行重大的努力，但GPT-4和Llama 3等大型语言模型(LLM)仍然容易受到越狱攻击，这些攻击可能会导致有害行为，包括由对抗性后缀触发的行为。在先前研究的基础上，我们假设这些对抗性后缀不仅仅是错误，而且可能代表可以主导LLM行为的特征。为了评估这一假设，我们进行了几个实验。首先，我们证明了良性特征可以有效地用作对抗性后缀，即，我们开发了一种特征提取方法来从良性数据集中提取与样本无关的后缀形式的特征，并表明这些后缀可以有效地危害安全对齐。其次，我们证明了越狱攻击产生的对抗性后缀可能包含有意义的特征，即在不同的提示后添加相同的后缀会导致响应表现出特定的特征。第三，我们表明，这种良性但危及安全的特征可以通过仅使用良性数据集进行微调来轻松引入，即即使在没有有害内容的情况下也可以。这突出了在训练数据中占主导地位的良性特征所构成的关键风险，并呼吁进一步研究以加强LLM的安全一致性。我们的代码和数据可在\url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.上获得



## **41. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2407.03876v2) [paper-pdf](http://arxiv.org/pdf/2407.03876v2)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **42. Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models**

利用任务过载对大型语言模型进行可扩展越狱攻击 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04190v1) [paper-pdf](http://arxiv.org/pdf/2410.04190v1)

**Authors**: Yiting Dong, Guobin Shen, Dongcheng Zhao, Xiang He, Yi Zeng

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass their safety mechanisms. Existing attack methods are fixed or specifically tailored for certain models and cannot flexibly adjust attack strength, which is critical for generalization when attacking models of various sizes. We introduce a novel scalable jailbreak attack that preempts the activation of an LLM's safety policies by occupying its computational resources. Our method involves engaging the LLM in a resource-intensive preliminary task - a Character Map lookup and decoding process - before presenting the target instruction. By saturating the model's processing capacity, we prevent the activation of safety protocols when processing the subsequent instruction. Extensive experiments on state-of-the-art LLMs demonstrate that our method achieves a high success rate in bypassing safety measures without requiring gradient access, manual prompt engineering. We verified our approach offers a scalable attack that quantifies attack strength and adapts to different model scales at the optimal strength. We shows safety policies of LLMs might be more susceptible to resource constraints. Our findings reveal a critical vulnerability in current LLM safety designs, highlighting the need for more robust defense strategies that account for resource-intense condition.

摘要: 大型语言模型(LLM)仍然容易受到绕过其安全机制的越狱攻击。现有的攻击方法是固定的或针对特定模型量身定做的，不能灵活调整攻击强度，这对于攻击不同大小的模型时的泛化至关重要。我们提出了一种新的可扩展的越狱攻击，该攻击通过占用LLM的计算资源来抢占LLM安全策略的激活。我们的方法包括在呈现目标指令之前，让LLM参与一个资源密集型的预备任务-字符映射查找和解码过程。通过使模型的处理能力饱和，我们防止在处理后续指令时激活安全协议。在最先进的LLMS上的广泛实验表明，我们的方法在绕过安全措施方面取得了很高的成功率，而不需要梯度访问、人工提示工程。我们验证了我们的方法提供了一种可扩展的攻击，它量化了攻击强度，并以最佳强度适应不同的模型规模。我们发现，低成本管理的安全政策可能更容易受到资源约束的影响。我们的发现揭示了当前LLM安全设计中的一个严重漏洞，突显了需要更强大的防御战略来应对资源密集型条件。



## **43. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

我们可以信任有保障的代理人吗？探索针对基于LLM的决策系统的后门攻击 cs.CR

31 pages, including main paper, references, and appendix

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2405.20774v2) [paper-pdf](http://arxiv.org/pdf/2405.20774v2)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.

摘要: 大型语言模型(LLM)在真实世界的人工智能决策任务中显示出了巨大的前景，特别是在进行微调以利用它们固有的常识和推理能力，同时为特定应用量身定做时。然而，这一微调过程引入了相当大的安全和安全漏洞，特别是在安全关键的网络物理系统中。在这项工作中，我们提出了第一个全面的框架，对基于LLM的决策系统(BALD)的后门攻击，系统地研究了攻击面和触发机制。具体地说，我们提出了三种不同的攻击机制：单词注入、场景操纵和知识注入，分别针对基于LLM的决策流水线中的各个组件。我们在自主驾驶和家用机器人任务中对具有代表性的LLM(GPT-3.5、LLaMA2、Palm2)进行了广泛的实验，展示了我们的后门触发器在各种攻击渠道中的有效性和隐蔽性，例如车辆加速驶向障碍物和机器人将刀放在床上。我们的单词和知识注入攻击在多个模型和数据集上实现了近100%的成功率，而只需要有限的系统访问权限。我们的场景操纵攻击的成功率超过65%，高达90%，并且不需要任何运行时系统入侵。我们还评估了这些攻击对防御的健壮性，揭示了它们的弹性。我们的发现突出了嵌入式LLM系统中的关键安全漏洞，并强调了保护这些系统以降低潜在风险的迫切需要。



## **44. ASPIRER: Bypassing System Prompts With Permutation-based Backdoors in LLMs**

ASPIRER：在LLM中使用基于置换的后门来确定询问系统 cs.CR

**SubmitDate**: 2024-10-05    [abs](http://arxiv.org/abs/2410.04009v1) [paper-pdf](http://arxiv.org/pdf/2410.04009v1)

**Authors**: Lu Yan, Siyuan Cheng, Xuan Chen, Kaiyuan Zhang, Guangyu Shen, Zhuo Zhang, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50% while maintaining a clean accuracy (CACC) of 98.58%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.

摘要: 大型语言模型(LLM)已经成为许多应用程序不可或缺的一部分，系统提示是规范模型行为和确保道德输出的关键机制。在本文中，我们引入了一种新型的后门攻击，它系统地绕过了这些系统提示，给人工智能供应链带来了重大风险。在正常情况下，模型严格遵循其系统提示。然而，我们的后门允许恶意行为者在触发时绕过这些安全措施。具体地说，我们将探讨LLM提供程序在基本模型中嵌入隐蔽触发器的场景。下游部署人员不知道隐藏的触发器，对模型进行微调，并将其作为服务提供给用户。恶意攻击者可以从提供商购买触发器，并使用它来利用已部署的模型，从而禁用系统提示并实现受限的结果。我们的攻击利用了置换触发器，只有当其组件按精确顺序排列时才会激活，这使得检测或反向工程在计算上具有挑战性。我们在五个最先进的模型上对我们的方法进行了评估，表明我们的方法实现了高达99.50%的攻击成功率(ASR)，同时保持了98.58%的干净准确率(CACC)，即使在防御微调之后也是如此。这些发现突显了LLM部署管道中的关键漏洞，并强调了加强防御的必要性。



## **45. Detecting Machine-Generated Long-Form Content with Latent-Space Variables**

检测具有潜在空间变量的机器生成的长形式内容 cs.CL

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03856v1) [paper-pdf](http://arxiv.org/pdf/2410.03856v1)

**Authors**: Yufei Tian, Zeyu Pan, Nanyun Peng

**Abstract**: The increasing capability of large language models (LLMs) to generate fluent long-form texts is presenting new challenges in distinguishing machine-generated outputs from human-written ones, which is crucial for ensuring authenticity and trustworthiness of expressions. Existing zero-shot detectors primarily focus on token-level distributions, which are vulnerable to real-world domain shifts, including different prompting and decoding strategies, and adversarial attacks. We propose a more robust method that incorporates abstract elements, such as event transitions, as key deciding factors to detect machine versus human texts by training a latent-space model on sequences of events or topics derived from human-written texts. In three different domains, machine-generated texts, which are originally inseparable from human texts on the token level, can be better distinguished with our latent-space model, leading to a 31% improvement over strong baselines such as DetectGPT. Our analysis further reveals that, unlike humans, modern LLMs like GPT-4 generate event triggers and their transitions differently, an inherent disparity that helps our method to robustly detect machine-generated texts.

摘要: 大型语言模型(LLM)生成流畅的长文本的能力日益增强，这对区分机器生成的输出和人类书写的输出提出了新的挑战，这对确保表达的真实性和可信度至关重要。现有的零射击检测器主要集中在令牌级分发上，这些分发容易受到现实世界域转换的影响，包括不同的提示和解码策略，以及敌意攻击。我们提出了一种更健壮的方法，通过对来自人类书写的文本的事件或主题序列训练潜在空间模型，将事件转移等抽象元素作为关键决定因素来检测机器文本与人类文本。在三个不同的领域中，机器生成的文本在标记级别上与人类文本密不可分，使用我们的潜在空间模型可以更好地区分它们，导致比DetectGPT等强基线提高31%。我们的分析进一步表明，与人类不同的是，像GPT-4这样的现代LLM以不同的方式生成事件触发器及其转换，这一固有的差异有助于我们的方法稳健地检测机器生成的文本。



## **46. Developing Assurance Cases for Adversarial Robustness and Regulatory Compliance in LLMs**

在LLC中开发对抗稳健性和监管合规性的保证案例 cs.CR

Accepted to the ASSURE 2024 workshop

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.05304v1) [paper-pdf](http://arxiv.org/pdf/2410.05304v1)

**Authors**: Tomas Bueno Momcilovic, Dian Balta, Beat Buesser, Giulio Zizzo, Mark Purcell

**Abstract**: This paper presents an approach to developing assurance cases for adversarial robustness and regulatory compliance in large language models (LLMs). Focusing on both natural and code language tasks, we explore the vulnerabilities these models face, including adversarial attacks based on jailbreaking, heuristics, and randomization. We propose a layered framework incorporating guardrails at various stages of LLM deployment, aimed at mitigating these attacks and ensuring compliance with the EU AI Act. Our approach includes a meta-layer for dynamic risk management and reasoning, crucial for addressing the evolving nature of LLM vulnerabilities. We illustrate our method with two exemplary assurance cases, highlighting how different contexts demand tailored strategies to ensure robust and compliant AI systems.

摘要: 本文提出了一种在大型语言模型（LLM）中开发对抗稳健性和监管合规性的保证案例的方法。我们重点关注自然语言和代码语言任务，探索这些模型面临的漏洞，包括基于越狱、启发式和随机化的对抗攻击。我们提出了一个分层框架，在LLM部署的各个阶段纳入护栏，旨在减轻这些攻击并确保遵守欧盟人工智能法案。我们的方法包括用于动态风险管理和推理的元层，这对于解决LLM漏洞不断变化的性质至关重要。我们通过两个示例性保证案例来说明我们的方法，强调不同的环境如何要求量身定制的策略来确保强大且合规的人工智能系统。



## **47. RAFT: Realistic Attacks to Fool Text Detectors**

RAFT：愚弄文本检测器的现实攻击 cs.CL

Accepted by EMNLP 2024

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03658v1) [paper-pdf](http://arxiv.org/pdf/2410.03658v1)

**Authors**: James Wang, Ran Li, Junfeng Yang, Chengzhi Mao

**Abstract**: Large language models (LLMs) have exhibited remarkable fluency across various tasks. However, their unethical applications, such as disseminating disinformation, have become a growing concern. Although recent works have proposed a number of LLM detection methods, their robustness and reliability remain unclear. In this paper, we present RAFT: a grammar error-free black-box attack against existing LLM detectors. In contrast to previous attacks for language models, our method exploits the transferability of LLM embeddings at the word-level while preserving the original text quality. We leverage an auxiliary embedding to greedily select candidate words to perturb against the target detector. Experiments reveal that our attack effectively compromises all detectors in the study across various domains by up to 99%, and are transferable across source models. Manual human evaluation studies show our attacks are realistic and indistinguishable from original human-written text. We also show that examples generated by RAFT can be used to train adversarially robust detectors. Our work shows that current LLM detectors are not adversarially robust, underscoring the urgent need for more resilient detection mechanisms.

摘要: 大型语言模型(LLM)在各种任务中表现出了惊人的流畅性。然而，它们不道德的应用，如传播虚假信息，已经成为一个日益令人担忧的问题。虽然最近的工作已经提出了一些LLM检测方法，但它们的稳健性和可靠性仍然不清楚。本文提出了一种针对现有LLM检测器的无语法错误的黑盒攻击方法RAFT。与以往对语言模型的攻击不同，我们的方法在保持原始文本质量的同时，利用了LLM嵌入在单词级别的可转移性。我们利用辅助嵌入来贪婪地选择候选单词来扰动目标检测器。实验表明，我们的攻击有效地危害了研究中跨不同域的所有检测器高达99%，并且可以跨源模型传输。人工人工评估研究表明，我们的攻击是真实的，与原始的人类书面文本没有什么区别。我们还表明，由RAFT生成的例子可以用于训练对抗性稳健的检测器。我们的工作表明，目前的LLM检测器并不具有相反的健壮性，这突显了对更具弹性的检测机制的迫切需要。



## **48. Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation**

系好安全带：通过数据修复在每个定制阶段对LLM进行优化 cs.CR

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.02220v2) [paper-pdf](http://arxiv.org/pdf/2410.02220v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Chenyu You, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are extensively adapted for downstream applications through a process known as "customization," with fine-tuning being a common method for integrating domain-specific expertise. However, recent studies have revealed a vulnerability that tuning LLMs with malicious samples can compromise their robustness and amplify harmful content, an attack known as "jailbreaking." To mitigate such attack, we propose an effective defensive framework utilizing data curation to revise commonsense texts and enhance their safety implication from the perspective of LLMs. The curated texts can mitigate jailbreaking attacks at every stage of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize jailbreaking risks, or after customization to restore the compromised models. Since the curated data strengthens LLMs through the standard fine-tuning workflow, we do not introduce additional modules during LLM inference, thereby preserving the original customization process. Experimental results demonstrate a substantial reduction in jailbreaking effects, with up to a 100% success in generating responsible responses. Notably, our method is effective even with commonsense texts, which are often more readily available than safety-relevant data. With the every-stage defensive framework and supporting experimental performance, this work represents a significant advancement in mitigating jailbreaking risks and ensuring the secure customization of LLMs.

摘要: 大型语言模型(LLM)通过一种称为“定制”的过程广泛适用于下游应用程序，微调是集成特定领域专业知识的常见方法。然而，最近的研究揭示了一个漏洞，即用恶意样本调整LLM可能会损害它们的健壮性，并放大有害内容，这种攻击被称为“越狱”。为了缓解这种攻击，我们提出了一个有效的防御框架，利用数据精选来修改常识文本，并从LLMS的角度增强其安全含义。经过精选的文本可以在定制过程的每个阶段减少越狱攻击：在定制之前，以使LLM免受未来的越狱企图；在定制期间，以中和越狱风险；或在定制之后，以恢复受影响的模型。由于精选数据通过标准的微调工作流程加强了LLM，因此我们在LLM推理过程中不会引入额外的模块，从而保留了原始的定制流程。实验结果表明，越狱效果大大降低，生成负责任的响应的成功率高达100%。值得注意的是，我们的方法甚至对于常识性文本也是有效的，这些常识性文本通常比安全相关数据更容易获得。通过每个阶段的防御框架和支持的实验性能，这项工作在降低越狱风险和确保低成本管理系统的安全定制方面取得了重大进展。



## **49. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2406.14393v3) [paper-pdf](http://arxiv.org/pdf/2406.14393v3)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

摘要: 大型语言模型(LLM)的广泛采用引起了人们对它们的安全性和可靠性的担忧，特别是它们对对手攻击的脆弱性。在本文中，我们提出了一种新的观点，将该漏洞归因于对齐过程中的错误指定。当奖励函数未能准确捕获预期行为时，就会出现这种错误说明，从而导致模型输出不对齐。我们引入了一个度量指标ReGap来量化奖励错误指定的程度，并展示了它在检测有害后门提示方面的有效性和健壮性。在这些见解的基础上，我们提出了REMISTY，这是一个用于自动红色团队的系统，它在错误指定奖励的空间中生成对抗性提示。在保持生成提示的人类可读性的同时，针对各种目标对齐的LLM，在AdvBtch基准上实现了最先进的攻击成功率。此外，这些对开源模型的攻击表明，可以很好地转移到GPT-4o等封闭源代码模型和来自HarmBtch的非分发任务。详细的分析强调了与以前的方法相比，所提出的奖励误指定目标的独特优势，为提高LLM的安全性和稳健性提供了新的见解。



## **50. AutoPenBench: Benchmarking Generative Agents for Penetration Testing**

AutoPenBench：渗透测试生成剂的基准测试 cs.CR

Codes for the benchmark:  https://github.com/lucagioacchini/auto-pen-bench Codes for the paper  experiments: https://github.com/lucagioacchini/genai-pentest-paper

**SubmitDate**: 2024-10-04    [abs](http://arxiv.org/abs/2410.03225v1) [paper-pdf](http://arxiv.org/pdf/2410.03225v1)

**Authors**: Luca Gioacchini, Marco Mellia, Idilio Drago, Alexander Delsanto, Giuseppe Siracusano, Roberto Bifulco

**Abstract**: Generative AI agents, software systems powered by Large Language Models (LLMs), are emerging as a promising approach to automate cybersecurity tasks. Among the others, penetration testing is a challenging field due to the task complexity and the diverse strategies to simulate cyber-attacks. Despite growing interest and initial studies in automating penetration testing with generative agents, there remains a significant gap in the form of a comprehensive and standard framework for their evaluation and development. This paper introduces AutoPenBench, an open benchmark for evaluating generative agents in automated penetration testing. We present a comprehensive framework that includes 33 tasks, each representing a vulnerable system that the agent has to attack. Tasks are of increasing difficulty levels, including in-vitro and real-world scenarios. We assess the agent performance with generic and specific milestones that allow us to compare results in a standardised manner and understand the limits of the agent under test. We show the benefits of AutoPenBench by testing two agent architectures: a fully autonomous and a semi-autonomous supporting human interaction. We compare their performance and limitations. For example, the fully autonomous agent performs unsatisfactorily achieving a 21% Success Rate (SR) across the benchmark, solving 27% of the simple tasks and only one real-world task. In contrast, the assisted agent demonstrates substantial improvements, with 64% of SR. AutoPenBench allows us also to observe how different LLMs like GPT-4o or OpenAI o1 impact the ability of the agents to complete the tasks. We believe that our benchmark fills the gap with a standard and flexible framework to compare penetration testing agents on a common ground. We hope to extend AutoPenBench along with the research community by making it available under https://github.com/lucagioacchini/auto-pen-bench.

摘要: 生成式人工智能代理是由大型语言模型(LLM)支持的软件系统，正在成为一种有前途的自动化网络安全任务的方法。其中，渗透测试是一个具有挑战性的领域，因为任务的复杂性和模拟网络攻击的策略多种多样。尽管人们对利用产生剂进行自动化渗透测试越来越感兴趣，并进行了初步研究，但在评估和开发产生剂的全面和标准框架的形式上，仍然存在着重大差距。本文介绍了一种用于评估自动渗透测试中的生成性代理的开放基准--AutoPenBch。我们提出了一个全面的框架，包括33个任务，每个任务代表代理必须攻击的易受攻击的系统。任务的难度越来越高，包括体外和真实世界的场景。我们用通用的和特定的里程碑来评估代理的性能，使我们能够以标准化的方式比较结果，并了解接受测试的代理的限制。我们通过测试两种代理体系结构：完全自主和半自主支持人类交互，展示了AutoPenB边的好处。我们比较了它们的性能和局限性。例如，完全自主的代理在基准测试中的成功率(SR)不令人满意地达到了21%，解决了27%的简单任务，而只解决了一个真实世界的任务。相比之下，辅助剂表现出显著的改善，获得了SR的5%。AutoPenB边还允许我们观察不同的LLM，如GPT-40或OpenAI o1，是如何影响代理完成任务的能力的。我们相信，我们的基准填补了这一空白，提供了一个标准和灵活的框架，可以在共同的基础上比较渗透测试试剂。我们希望通过使其在https://github.com/lucagioacchini/auto-pen-bench.下可用来与研究社区一起扩展AutoPenB边



