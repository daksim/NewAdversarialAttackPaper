# Latest Large Language Model Attack Papers
**update at 2024-08-27 18:56:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Investigating the Effectiveness of Bayesian Spam Filters in Detecting LLM-modified Spam Mails**

调查Bayesian垃圾邮件过滤器检测LLM修改的垃圾邮件的有效性 cs.CR

EAI International Conference on Digital Forensics & Cyber Crime 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14293v1) [paper-pdf](http://arxiv.org/pdf/2408.14293v1)

**Authors**: Malte Josten, Torben Weis

**Abstract**: Spam and phishing remain critical threats in cybersecurity, responsible for nearly 90% of security incidents. As these attacks grow in sophistication, the need for robust defensive mechanisms intensifies. Bayesian spam filters, like the widely adopted open-source SpamAssassin, are essential tools in this fight. However, the emergence of large language models (LLMs) such as ChatGPT presents new challenges. These models are not only powerful and accessible, but also inexpensive to use, raising concerns about their misuse in crafting sophisticated spam emails that evade traditional spam filters. This work aims to evaluate the robustness and effectiveness of SpamAssassin against LLM-modified email content. We developed a pipeline to test this vulnerability. Our pipeline modifies spam emails using GPT-3.5 Turbo and assesses SpamAssassin's ability to classify these modified emails correctly. The results show that SpamAssassin misclassified up to 73.7% of LLM-modified spam emails as legitimate. In contrast, a simpler dictionary-replacement attack showed a maximum success rate of only 0.4%. These findings highlight the significant threat posed by LLM-modified spam, especially given the cost-efficiency of such attacks (0.17 cents per email). This paper provides crucial insights into the vulnerabilities of current spam filters and the need for continuous improvement in cybersecurity measures.

摘要: 垃圾邮件和网络钓鱼仍然是网络安全中的关键威胁，导致了近90%的安全事件。随着这些攻击变得越来越复杂，对强大防御机制的需求也变得更加迫切。贝叶斯垃圾邮件过滤器，就像被广泛采用的开源SpamAssassin一样，是这场斗争中必不可少的工具。然而，像ChatGPT这样的大型语言模型(LLM)的出现带来了新的挑战。这些模型不仅功能强大、易于访问，而且使用起来也不贵，这引发了人们对它们在制作复杂的垃圾邮件时被滥用的担忧，这些垃圾邮件绕过了传统的垃圾邮件过滤器。这项工作的目的是评估SpamAssassin对LLM修改的电子邮件内容的健壮性和有效性。我们开发了一条管道来测试这个漏洞。我们的渠道使用GPT-3.5 Turbo修改垃圾电子邮件，并评估SpamAssassin对这些修改后的电子邮件进行正确分类的能力。结果表明，Spamassassin将高达73.7%的LLM修改后的垃圾邮件错误分类为合法邮件。相比之下，更简单的词典替换攻击的最高成功率仅为0.4%。这些发现突显了LLM修改的垃圾邮件构成的重大威胁，特别是考虑到此类攻击的成本效益(每封电子邮件0.17美分)。本文对当前垃圾邮件过滤器的漏洞以及网络安全措施持续改进的必要性提供了至关重要的见解。



## **2. Beyond Detection: Leveraging Large Language Models for Cyber Attack Prediction in IoT Networks**

超越检测：利用大型语言模型进行物联网网络中的网络攻击预测 cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14045v1) [paper-pdf](http://arxiv.org/pdf/2408.14045v1)

**Authors**: Alaeddine Diaf, Abdelaziz Amara Korba, Nour Elislem Karabadji, Yacine Ghamri-Doudane

**Abstract**: In recent years, numerous large-scale cyberattacks have exploited Internet of Things (IoT) devices, a phenomenon that is expected to escalate with the continuing proliferation of IoT technology. Despite considerable efforts in attack detection, intrusion detection systems remain mostly reactive, responding to specific patterns or observed anomalies. This work proposes a proactive approach to anticipate and mitigate malicious activities before they cause damage. This paper proposes a novel network intrusion prediction framework that combines Large Language Models (LLMs) with Long Short Term Memory (LSTM) networks. The framework incorporates two LLMs in a feedback loop: a fine-tuned Generative Pre-trained Transformer (GPT) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) for evaluating the predicted traffic. The LSTM classifier model then identifies malicious packets among these predictions. Our framework, evaluated on the CICIoT2023 IoT attack dataset, demonstrates a significant improvement in predictive capabilities, achieving an overall accuracy of 98%, offering a robust solution to IoT cybersecurity challenges.

摘要: 近年来，大量大规模网络攻击利用了物联网(IoT)设备，预计随着物联网技术的持续扩散，这一现象将升级。尽管在攻击检测方面做出了相当大的努力，入侵检测系统仍然主要是被动的，对特定的模式或观察到的异常做出反应。这项工作提出了一种主动的方法，在恶意活动造成破坏之前对其进行预测和缓解。提出了一种将大语言模型和长短期记忆网络相结合的网络入侵预测框架。该框架在反馈环路中结合了两个LLM：用于预测网络流量的微调生成式预训练变压器(GPT)模型和用于评估预测流量的来自变压器的微调双向编码器表示(BERT)。然后，LSTM分类器模型在这些预测中识别恶意数据包。我们的框架在CICIoT2023物联网攻击数据集上进行了评估，显示出预测能力的显著改进，总体准确率达到98%，为物联网网络安全挑战提供了强大的解决方案。



## **3. Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation**

通过不安全解码路径生成探索大型语言模型的安全响应边界 cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.10668v3) [paper-pdf](http://arxiv.org/pdf/2408.10668v3)

**Authors**: Haoyu Wang, Bingzhe Wu, Yatao Bian, Yongzhe Chang, Xueqian Wang, Peilin Zhao

**Abstract**: Large Language Models (LLMs) are implicit troublemakers. While they provide valuable insights and assist in problem-solving, they can also potentially serve as a resource for malicious activities. Implementing safety alignment could mitigate the risk of LLMs generating harmful responses. We argue that: even when an LLM appears to successfully block harmful queries, there may still be hidden vulnerabilities that could act as ticking time bombs. To identify these underlying weaknesses, we propose to use a cost value model as both a detector and an attacker. Trained on external or self-generated harmful datasets, the cost value model could successfully influence the original safe LLM to output toxic content in decoding process. For instance, LLaMA-2-chat 7B outputs 39.18% concrete toxic content, along with only 22.16% refusals without any harmful suffixes. These potential weaknesses can then be exploited via prompt optimization such as soft prompts on images. We name this decoding strategy: Jailbreak Value Decoding (JVD), emphasizing that seemingly secure LLMs may not be as safe as we initially believe. They could be used to gather harmful data or launch covert attacks.

摘要: 大型语言模型(LLM)是隐含的麻烦制造者。虽然它们提供了有价值的见解并帮助解决问题，但它们也可能成为恶意活动的来源。实施安全调整可以降低低密度脂蛋白产生有害反应的风险。我们认为：即使LLM似乎成功阻止了有害查询，仍可能存在隐藏的漏洞，这些漏洞可能会充当定时炸弹。为了识别这些潜在的弱点，我们建议使用成本价值模型作为检测器和攻击者。代价值模型在外部或自身产生的有害数据集上进行训练，可以成功地影响原始安全LLM在解码过程中输出有毒内容。例如，骆驼-2-Chat 7B输出39.18%的具体有毒内容，以及只有22.16%的拒绝没有任何有害的后缀。然后可以通过提示优化(如图像上的软提示)来利用这些潜在的弱点。我们将这种解码策略命名为：越狱价值解码(JVD)，强调看似安全的LLM可能并不像我们最初认为的那样安全。它们可能被用来收集有害数据或发动秘密攻击。



## **4. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

TF攻击：对大型语言模型的可转移且快速对抗攻击 cs.CL

14 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2305.17440 by other authors

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.13985v1) [paper-pdf](http://arxiv.org/pdf/2408.13985v1)

**Authors**: Zelin Li, Kehai Chen, Xuefeng Bai, Lemao Liu, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.

摘要: 近年来，随着大型语言模型的发展，针对大型语言模型的对抗性攻击引起了越来越多的关注。我们发现，现有的对抗性攻击方法表现出有限的可转移性和显著的低效，特别是当应用于LLM时。本文分析了以往主流对抗性攻击方法的核心机制，发现1)不同受害者模型的重要性分数分布明显不同，限制了可转移性；2)顺序攻击过程导致了大量的时间开销。基于以上两点，我们提出了一种新的方案，称为TF-Attack，用于对LLMS进行可转移和快速对抗攻击。TF-Attack使用外部LLM作为第三方监督者，而不是受害者模型来识别判刑内的关键单元。此外，TF-Attack还引入了重要度的概念，允许并行替换攻击。我们在6个广泛采用的基准上进行了广泛的实验，从自动度量和人工度量两个方面对所提出的方法进行了评估。结果表明，我们的方法在可转移性上始终优于以前的方法，并提供了显著的速度改进，比以前的攻击策略快20倍。



## **5. Large Language Models as Carriers of Hidden Messages**

大型语言模型作为隐藏消息的载体 cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2406.02481v3) [paper-pdf](http://arxiv.org/pdf/2406.02481v3)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: With the help of simple fine-tuning, one can artificially embed hidden text into large language models (LLMs). This text is revealed only when triggered by a specific query to the LLM. Two primary applications are LLM fingerprinting and steganography. In the context of LLM fingerprinting, a unique text identifier (fingerprint) is embedded within the model to verify licensing compliance. In the context of steganography, the LLM serves as a carrier for hidden messages that can be disclosed through a chosen trigger question.   Our work demonstrates that embedding hidden text in the LLM via fine-tuning, though seemingly secure due to the vast number of potential triggers (any sequence of characters or tokens could serve as a trigger), is susceptible to extraction through analysis of the LLM's output decoding process. We propose an extraction attack called Unconditional Token Forcing (UTF). It is premised on the hypothesis that iteratively feeding each token from the LLM's vocabulary into the model should reveal output sequences with abnormally high token probabilities, indicating potential hidden text candidates. We also present a defense method to hide text in such a way that it is resistant to both UTF and attacks based on sampling decoding methods, which we named Unconditional Token Forcing Confusion (UTFC). To the best of our knowledge, there is no attack method that can extract text hidden with UTFC. UTFC has both benign applications (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels).

摘要: 在简单微调的帮助下，人们可以人为地将隐藏文本嵌入到大型语言模型(LLM)中。只有在对LLM的特定查询触发时，才会显示此文本。两个主要应用是LLM指纹识别和隐写。在LLM指纹识别的上下文中，唯一的文本识别符(指纹)被嵌入到模型中，以验证许可合规性。在隐写术的背景下，LLM充当了隐藏消息的载体，这些隐藏消息可以通过选择的触发问题来泄露。我们的工作表明，通过微调将隐藏文本嵌入到LLM中，尽管由于潜在触发器(任何字符或标记序列都可以作为触发器)的数量巨大而看起来是安全的，但通过分析LLM的输出解码过程，它容易被提取。我们提出了一种称为无条件令牌强迫(UTF)的提取攻击。它的前提是假设迭代地将LLM词汇中的每个标记输入到模型中，应该会揭示出具有异常高的标记概率的输出序列，这表明潜在的隐藏文本候选。我们还提出了一种基于抽样解码的文本隐藏方法，称为无条件令牌强制混淆(UTFC)，使其能够同时抵抗UTF和攻击。就我们所知，没有一种攻击方法可以提取使用UTFC隐藏的文本。UTFC既有良性应用程序(改进LLM指纹识别)，也有恶意应用程序(使用LLM创建秘密通信通道)。



## **6. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2403.17710v2) [paper-pdf](http://arxiv.org/pdf/2403.17710v2)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies.

摘要: LLM-as-a-Court使用大型语言模型(LLM)从给定问题的一组候选人中选择最佳答案。LLM-as-a-Court有许多应用，如LLM支持的搜索、带人工智能反馈的强化学习(RLAIF)和工具选择。在这项工作中，我们提出了一种针对LLM-as-a-Court的基于优化的快速注入攻击--JudgeDeceiver。JudgeDeceiver将精心制作的序列注入到攻击者控制的候选响应中，以便LLM-as-a-Court为攻击者选择的问题选择候选响应，而不管其他候选响应是什么。具体地说，我们将寻找这样的序列描述为一个优化问题，并提出了一种基于梯度的方法来近似求解它。我们的广泛评估表明，JudgeDecept是非常有效的，并且比现有的手动手工创建注入序列的即时注入攻击和越狱攻击更有效，当扩展到我们的问题时。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了防御措施，包括已知答案检测、困惑检测和困惑加窗检测。我们的结果表明，这些防御措施是不够的，这突显了开发新的防御战略的迫切需要。



## **7. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

保护视觉语言模型免受修补视觉提示注入器的影响 cs.CV

15 pages

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.10529v2) [paper-pdf](http://arxiv.org/pdf/2405.10529v2)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.

摘要: 大型语言模型已变得越来越突出，这也标志着向多通道的转变，成为人工智能的下一个前沿，在人工智能中，它们的嵌入被用作生成文本内容的提示。视觉语言模型(VLM)站在这一进步的前沿，提供了将视觉和文本数据相结合的创新方法，以增强理解和交互。然而，这种整合也扩大了攻击面。基于补丁的对抗性攻击被认为是物理视觉应用中最现实的威胁模型，许多现有的文献都证明了这一点。在本文中，我们建议解决补丁视觉提示注入，即攻击者利用敌意补丁来生成VLMS中的目标内容。我们的调查显示，打补丁的对抗性提示显示出对像素随机化的敏感性，这一特征即使在旨在对抗此类防御的适应性攻击中也保持健壮。利用这一见解，我们推出了SmoothVLM，这是一种植根于平滑技术的防御机制，专门为保护VLM免受修补的视觉提示注入器的威胁而量身定做。我们的框架将攻击成功率显著降低到了0%到5.0%之间，同时实现了良性映像的67.3%到95.0%的上下文恢复，展示了安全性和可用性之间的平衡。



## **8. Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach**

探索视觉语言预训练模型的鲁棒性：多模式对抗攻击方法 cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2408.13461v1) [paper-pdf](http://arxiv.org/pdf/2408.13461v1)

**Authors**: Jiwei Guan, Tianyu Ding, Longbing Cao, Lei Pan, Chen Wang, Xi Zheng

**Abstract**: Vision-language pretraining (VLP) with transformers has demonstrated exceptional performance across numerous multimodal tasks. However, the adversarial robustness of these models has not been thoroughly investigated. Existing multimodal attack methods have largely overlooked cross-modal interactions between visual and textual modalities, particularly in the context of cross-attention mechanisms. In this paper, we study the adversarial vulnerability of recent VLP transformers and design a novel Joint Multimodal Transformer Feature Attack (JMTFA) that concurrently introduces adversarial perturbations in both visual and textual modalities under white-box settings. JMTFA strategically targets attention relevance scores to disrupt important features within each modality, generating adversarial samples by fusing perturbations and leading to erroneous model predictions. Experimental results indicate that the proposed approach achieves high attack success rates on vision-language understanding and reasoning downstream tasks compared to existing baselines. Notably, our findings reveal that the textual modality significantly influences the complex fusion processes within VLP transformers. Moreover, we observe no apparent relationship between model size and adversarial robustness under our proposed attacks. These insights emphasize a new dimension of adversarial robustness and underscore potential risks in the reliable deployment of multimodal AI systems.

摘要: 使用变压器的视觉语言预培训(VLP)在许多多模式任务中表现出了出色的性能。然而，这些模型的对抗稳健性还没有得到彻底的研究。现有的多通道攻击方法在很大程度上忽略了视觉通道和文本通道之间的跨通道交互作用，特别是在交叉注意机制的背景下。本文研究了现有VLP变换的对抗性漏洞，设计了一种在白盒环境下同时引入对抗性扰动的联合多模式变换特征攻击(JMTFA)。JMTFA战略性地将注意力相关性分数作为目标，以扰乱每个通道中的重要特征，通过融合扰动生成对抗性样本，并导致错误的模型预测。实验结果表明，与现有的基线相比，该方法在视觉语言理解和推理的下游任务上获得了更高的攻击成功率。值得注意的是，我们的研究结果显示，语篇情态显著影响VLP转换器内复杂的融合过程。此外，在我们提出的攻击下，我们没有观察到模型大小和对手稳健性之间的明显关系。这些见解强调了对抗性稳健性的一个新维度，并强调了可靠部署多模式人工智能系统的潜在风险。



## **9. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2407.14573v4) [paper-pdf](http://arxiv.org/pdf/2407.14573v4)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **10. Is Generative AI the Next Tactical Cyber Weapon For Threat Actors? Unforeseen Implications of AI Generated Cyber Attacks**

生成性人工智能是威胁行为者的下一个战术网络武器吗？人工智能引发的网络攻击的不可预见影响 cs.CR

Journal Paper

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12806v1) [paper-pdf](http://arxiv.org/pdf/2408.12806v1)

**Authors**: Yusuf Usman, Aadesh Upadhyay, Prashnna Gyawali, Robin Chataut

**Abstract**: In an era where digital threats are increasingly sophisticated, the intersection of Artificial Intelligence and cybersecurity presents both promising defenses and potent dangers. This paper delves into the escalating threat posed by the misuse of AI, specifically through the use of Large Language Models (LLMs). This study details various techniques like the switch method and character play method, which can be exploited by cybercriminals to generate and automate cyber attacks. Through a series of controlled experiments, the paper demonstrates how these models can be manipulated to bypass ethical and privacy safeguards to effectively generate cyber attacks such as social engineering, malicious code, payload generation, and spyware. By testing these AI generated attacks on live systems, the study assesses their effectiveness and the vulnerabilities they exploit, offering a practical perspective on the risks AI poses to critical infrastructure. We also introduce Occupy AI, a customized, finetuned LLM specifically engineered to automate and execute cyberattacks. This specialized AI driven tool is adept at crafting steps and generating executable code for a variety of cyber threats, including phishing, malware injection, and system exploitation. The results underscore the urgency for ethical AI practices, robust cybersecurity measures, and regulatory oversight to mitigate AI related threats. This paper aims to elevate awareness within the cybersecurity community about the evolving digital threat landscape, advocating for proactive defense strategies and responsible AI development to protect against emerging cyber threats.

摘要: 在一个数字威胁日益复杂的时代，人工智能和网络安全的交集既带来了有希望的防御，也带来了潜在的危险。本文深入研究了滥用人工智能带来的不断升级的威胁，特别是通过使用大型语言模型(LLM)。这项研究详细介绍了各种技术，如切换方法和角色扮演方法，网络犯罪分子可以利用这些技术来生成网络攻击并使其自动化。通过一系列受控实验，本文演示了如何操纵这些模型以绕过伦理和隐私保护措施，从而有效地生成网络攻击，如社会工程、恶意代码、有效负载生成和间谍软件。通过测试这些人工智能对实时系统的攻击，该研究评估了它们的有效性和它们利用的漏洞，为人工智能对关键基础设施构成的风险提供了一个实用的视角。我们还推出了占领AI，这是一款定制的、经过精细调整的LLM，专门设计用于自动化和执行网络攻击。这个专门的人工智能驱动工具擅长为各种网络威胁制作步骤和生成可执行代码，包括网络钓鱼、恶意软件注入和系统利用。这一结果突显了伦理人工智能做法、强有力的网络安全措施和监管监督的紧迫性，以缓解与人工智能相关的威胁。本文旨在提高网络安全社区对不断发展的数字威胁格局的认识，倡导积极主动的防御战略和负责任的人工智能发展，以防御新出现的网络威胁。



## **11. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models**

BackdoorLLM：大型语言模型后门攻击的综合基准 cs.AI

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12798v1) [paper-pdf](http://arxiv.org/pdf/2408.12798v1)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative Large Language Models (LLMs) have made significant strides across various tasks, but they remain vulnerable to backdoor attacks, where specific triggers in the prompt cause the LLM to generate adversary-desired responses. While most backdoor research has focused on vision or text classification tasks, backdoor attacks in text generation have been largely overlooked. In this work, we introduce \textit{BackdoorLLM}, the first comprehensive benchmark for studying backdoor attacks on LLMs. \textit{BackdoorLLM} features: 1) a repository of backdoor benchmarks with a standardized training pipeline, 2) diverse attack strategies, including data poisoning, weight poisoning, hidden state attacks, and chain-of-thought attacks, 3) extensive evaluations with over 200 experiments on 8 attacks across 7 scenarios and 6 model architectures, and 4) key insights into the effectiveness and limitations of backdoors in LLMs. We hope \textit{BackdoorLLM} will raise awareness of backdoor threats and contribute to advancing AI safety. The code is available at \url{https://github.com/bboylyg/BackdoorLLM}.

摘要: 生成性大型语言模型(LLM)已经在各种任务中取得了重大进展，但它们仍然容易受到后门攻击，在后门攻击中，提示中的特定触发器会导致LLM生成对手想要的响应。虽然大多数后门研究都集中在视觉或文本分类任务上，但文本生成中的后门攻击在很大程度上被忽视了。在这项工作中，我们介绍了第一个用于研究对LLM的后门攻击的全面基准测试。\textit{Backdoor LLM}的特点是：1)具有标准化培训管道的后门基准存储库；2)多样化的攻击策略，包括数据中毒、重量中毒、隐藏状态攻击和思想链攻击；3)对7个场景和6个模型架构中的8个攻击进行了200多个实验的广泛评估；4)对LLMS中后门的有效性和局限性的关键洞察。我们希望\textit{Backdoor LLM}将提高人们对后门威胁的认识，并为推进人工智能安全做出贡献。代码可在\url{https://github.com/bboylyg/BackdoorLLM}.



## **12. Can Large Language Models Automatically Jailbreak GPT-4V?**

大型语言模型可以自动越狱GPT-4V吗？ cs.CL

TrustNLP@NAACL2024 (Fourth Workshop on Trustworthy Natural Language  Processing)

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2407.16686v2) [paper-pdf](http://arxiv.org/pdf/2407.16686v2)

**Authors**: Yuanwei Wu, Yue Huang, Yixin Liu, Xiang Li, Pan Zhou, Lichao Sun

**Abstract**: GPT-4V has attracted considerable attention due to its extraordinary capacity for integrating and processing multimodal information. At the same time, its ability of face recognition raises new safety concerns of privacy leakage. Despite researchers' efforts in safety alignment through RLHF or preprocessing filters, vulnerabilities might still be exploited. In our study, we introduce AutoJailbreak, an innovative automatic jailbreak technique inspired by prompt optimization. We leverage Large Language Models (LLMs) for red-teaming to refine the jailbreak prompt and employ weak-to-strong in-context learning prompts to boost efficiency. Furthermore, we present an effective search method that incorporates early stopping to minimize optimization time and token expenditure. Our experiments demonstrate that AutoJailbreak significantly surpasses conventional methods, achieving an Attack Success Rate (ASR) exceeding 95.3\%. This research sheds light on strengthening GPT-4V security, underscoring the potential for LLMs to be exploited in compromising GPT-4V integrity.

摘要: GPT-4V由于其综合和处理多模式信息的非凡能力而引起了相当大的关注。与此同时，它的人脸识别能力引发了新的隐私泄露的安全担忧。尽管研究人员通过RLHF或预处理过滤器在安全匹配方面做出了努力，但漏洞仍有可能被利用。在我们的研究中，我们介绍了AutoJailBreak，这是一种受即时优化启发的创新的自动越狱技术。我们利用用于红色团队的大型语言模型(LLM)来改进越狱提示，并采用从弱到强的上下文学习提示来提高效率。此外，我们还提出了一种结合提前停止的有效搜索方法，以最小化优化时间和令牌开销。实验表明，AutoJailBreak的攻击成功率(ASR)超过95.3%，明显优于传统方法。这项研究有助于加强GPT-4V的安全性，强调了LLMS在危害GPT-4V完整性方面的潜力。



## **13. LLM-PBE: Assessing Data Privacy in Large Language Models**

LLM-PBE：评估大型语言模型中的数据隐私 cs.CR

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12787v1) [paper-pdf](http://arxiv.org/pdf/2408.12787v1)

**Authors**: Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, Dawn Song

**Abstract**: Large Language Models (LLMs) have become integral to numerous domains, significantly advancing applications in data management, mining, and analysis. Their profound capabilities in processing and interpreting complex language data, however, bring to light pressing concerns regarding data privacy, especially the risk of unintentional training data leakage. Despite the critical nature of this issue, there has been no existing literature to offer a comprehensive assessment of data privacy risks in LLMs. Addressing this gap, our paper introduces LLM-PBE, a toolkit crafted specifically for the systematic evaluation of data privacy risks in LLMs. LLM-PBE is designed to analyze privacy across the entire lifecycle of LLMs, incorporating diverse attack and defense strategies, and handling various data types and metrics. Through detailed experimentation with multiple LLMs, LLM-PBE facilitates an in-depth exploration of data privacy concerns, shedding light on influential factors such as model size, data characteristics, and evolving temporal dimensions. This study not only enriches the understanding of privacy issues in LLMs but also serves as a vital resource for future research in the field. Aimed at enhancing the breadth of knowledge in this area, the findings, resources, and our full technical report are made available at https://llm-pbe.github.io/, providing an open platform for academic and practical advancements in LLM privacy assessment.

摘要: 大型语言模型(LLM)已经成为许多领域不可或缺的一部分，极大地推动了数据管理、挖掘和分析方面的应用。然而，它们在处理和解释复杂语言数据方面的深厚能力暴露了人们对数据隐私的迫切关切，特别是无意中泄露培训数据的风险。尽管这一问题具有严重的性质，但目前还没有文献对低成本管理中的数据隐私风险进行全面评估。针对这一差距，我们引入了LLM-PBE，这是一个专门为系统评估LLMS中的数据隐私风险而设计的工具包。LLm-PBE旨在分析LLMS整个生命周期中的隐私，整合不同的攻击和防御策略，并处理各种数据类型和指标。通过对多个LLM的详细实验，LLM-PBE有助于深入探索数据隐私问题，揭示模型大小、数据特征和不断演变的时间维度等影响因素。这一研究不仅丰富了对LLMS中隐私问题的理解，也为该领域未来的研究提供了重要的资源。为了提高这一领域的知识广度，我们的调查结果、资源和完整的技术报告可在https://llm-pbe.github.io/，上获得，为LLM隐私评估的学术和实践进步提供了一个开放的平台。



## **14. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

前置指导：大型语言模型防御越狱攻击的方向盘 cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.08924v2) [paper-pdf](http://arxiv.org/pdf/2408.08924v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance. our code is available at https://github.com/weiyezhimeng/Prefix-Guidance.

摘要: 近年来，大型语言模型的快速发展在各种任务中取得了显著的性能。然而，研究表明，LLMS容易受到越狱攻击，在越狱攻击中，攻击者可以通过精心制作的提示来诱导生成有害内容。此漏洞对安全使用和推广LLMS构成重大挑战。现有的防御方法从不同的角度提供保护，但往往存在有效性不足或对模型能力产生重大影响的问题。本文提出了一种即插即用、易于部署的越狱防御框架--前缀引导(PG)，它通过直接设置模型输出的前几个令牌来引导模型识别有害提示。这种方法将模型固有的安全功能与外部分类器相结合，以防御越狱攻击。我们在三个模型和五种攻击方法上演示了PG的有效性。与基线相比，我们的方法总体上更有效。此外，在Just-Eval基准上的结果进一步证实了PG在保持模型性能方面的优越性。我们的代码可以在https://github.com/weiyezhimeng/Prefix-Guidance.上找到



## **15. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

函数调用的阴暗面：越狱大型语言模型的途径 cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2407.17915v2) [paper-pdf](http://arxiv.org/pdf/2407.17915v2)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.

摘要: 大型语言模型(LLM)已经展示了非凡的能力，但它们的强大也伴随着重要的安全考虑。虽然已经对聊天模式下的LLMS的安全性进行了广泛的研究，但其函数调用功能的安全含义在很大程度上被忽视了。本文揭示了LLMS函数调用过程中的一个严重漏洞，引入了一种新的“越狱函数”攻击方法，该方法利用了对齐差异、用户胁迫和缺乏严格的安全过滤器。我们在包括GPT-40、Claude-3.5-Sonnet和Gemini-1.5-Pro在内的六个最先进的LLM上进行的经验研究显示，该攻击的平均成功率超过90%，这是令人震惊的。我们对函数调用容易受到此类攻击的原因进行了全面分析，并提出了防御策略，包括使用防御提示。我们的发现突显了在LLMS的函数调用能力方面迫切需要增强安全措施，通过识别以前未探索的风险、设计有效的攻击方法并提出实用的防御措施来促进人工智能安全领域。我们的代码可以在https://github.com/wooozihui/jailbreakfunction.上找到



## **16. Vaccine: Perturbation-aware Alignment for Large Language Models against Harmful Fine-tuning**

疫苗：大型语言模型的扰动感知对齐，防止有害的微调 cs.LG

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2402.01109v4) [paper-pdf](http://arxiv.org/pdf/2402.01109v4)

**Authors**: Tiansheng Huang, Sihao Hu, Ling Liu

**Abstract**: The new paradigm of finetuning-as-a-service introduces a new attack surface for Large Language Models (LLMs): a few harmful data uploaded by users can easily trick the finetuning to produce an alignment-broken model. We conduct an empirical analysis and uncover a \textit{harmful embedding drift} phenomenon, showing a probable cause of the alignment-broken effect. Inspired by our findings, we propose Vaccine, a perturbation-aware alignment technique to mitigate the security risk of users finetuning. The core idea of Vaccine is to produce invariant hidden embeddings by progressively adding crafted perturbation to them in the alignment phase. This enables the embeddings to withstand harmful perturbation from un-sanitized user data in the finetuning phase. Our results on open source mainstream LLMs (e.g., Llama2, Opt, Vicuna) demonstrate that Vaccine can boost the robustness of alignment against harmful prompts induced embedding drift while reserving reasoning ability towards benign prompts. Our code is available at \url{https://github.com/git-disl/Vaccine}.

摘要: 精调即服务的新范式为大型语言模型(LLM)引入了一个新的攻击面：用户上传的少量有害数据就可以很容易地欺骗精调，产生一个破坏对齐的模型。我们进行了实证分析，发现了一种有害的嵌入漂移现象，揭示了排列断裂效应的可能原因。受我们发现的启发，我们提出了Vaccine，一种扰动感知的对齐技术，以降低用户精调的安全风险。Vaccine的核心思想是通过在比对阶段逐步向其添加精心制作的扰动来产生不变的隐藏嵌入。这使嵌入能够在精细调整阶段抵御来自未清理的用户数据的有害干扰。我们在开源主流LLMS(如Llama2、Opt、Vicuna)上的实验结果表明，疫苗可以提高对有害提示导致的嵌入漂移的健壮性，同时保留对良性提示的推理能力。我们的代码可在\url{https://github.com/git-disl/Vaccine}.



## **17. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

隐性对抗培训提高了法学硕士对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.15549v2) [paper-pdf](http://arxiv.org/pdf/2407.15549v2)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，LLM的红队文献已经创造了各种各样的“越狱”技术，从经过微调的无害的模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **18. A Study of Backdoors in Instruction Fine-tuned Language Models**

微调语言模型教学中的后门研究 cs.CR

Under review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.07778v2) [paper-pdf](http://arxiv.org/pdf/2406.07778v2)

**Authors**: Jayaram Raghuram, George Kesidis, David J. Miller

**Abstract**: Backdoor data poisoning, inserted within instruction examples used to fine-tune a foundation Large Language Model (LLM) for downstream tasks (\textit{e.g.,} sentiment prediction), is a serious security concern due to the evasive nature of such attacks. The poisoning is usually in the form of a (seemingly innocuous) trigger word or phrase inserted into a very small fraction of the fine-tuning samples from a target class. Such backdoor attacks can: alter response sentiment, violate censorship, over-refuse (invoke censorship for legitimate queries), inject false content, or trigger nonsense responses (hallucinations). In this work we investigate the efficacy of instruction fine-tuning backdoor attacks as attack "hyperparameters" are varied under a variety of scenarios, considering: the trigger location in the poisoned examples; robustness to change in the trigger location, partial triggers, and synonym substitutions at test time; attack transfer from one (fine-tuning) domain to a related test domain; and clean-label vs. dirty-label poisoning. Based on our observations, we propose and evaluate two defenses against these attacks: i) a \textit{during-fine-tuning defense} based on word-frequency counts that assumes the (possibly poisoned) fine-tuning dataset is available and identifies the backdoor trigger tokens; and ii) a \textit{post-fine-tuning defense} based on downstream clean fine-tuning of the backdoored LLM with a small defense dataset. Finally, we provide a brief survey of related work on backdoor attacks and defenses.

摘要: 由于此类攻击的规避性质，后门数据中毒是一个严重的安全问题，它被插入到用于微调下游任务的基础大型语言模型(LLM)的指令示例中(例如，情感预测)。中毒通常以(看似无害的)触发词或短语的形式插入到来自目标类的微调样本的非常小的一部分中。这种后门攻击可以：改变回应情绪、违反审查制度、过度拒绝(对合法查询调用审查制度)、注入虚假内容或引发无稽之谈的反应(幻觉)。在这项工作中，我们研究了指令微调后门攻击的有效性，因为攻击“超参数”在各种场景下是不同的，考虑到：中毒示例中的触发器位置；对测试时触发器位置、部分触发器和同义词替换的稳健性；攻击从一个(微调)域转移到相关测试域；以及干净标签与脏标签中毒。基于我们的观察，我们提出并评估了针对这些攻击的两种防御方案：i)基于词频计数的精调期间防御方案，该方案假定(可能有毒的)微调数据集可用，并识别后门触发令牌；以及ii)基于带有小防御数据集的后置LLM的下游干净微调的后门微调防御方案。最后，我们对后门攻击和防御的相关工作进行了简要的概述。



## **19. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2402.06255v3) [paper-pdf](http://arxiv.org/pdf/2402.06255v3)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在有害内容过滤或启发式防御提示设计上。然而，如何通过提示实现内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对灰盒攻击和黑盒攻击都是有效的，在保持模型对良性任务的实用性的同时，将高级攻击的成功率降低到接近0。所提出的防御策略只需要很少的计算开销，为未来在LLM安全方面的探索开辟了新的前景。我们的代码可以在https://github.com/rain152/PAT.上找到



## **20. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.

摘要: 尽管最近大型的预训练神经语言模型(LLM)取得了成功，但人们对它们在预训练中学习的语言结构的表征知之甚少，这可能会导致对迅速变化或分布变化的意外行为。为了更好地理解这些模型和行为，我们引入了一个通用的模型分析框架，从它们对人类可解释的语言属性的表示和使用方面来研究LLM。基于能力的语言模型分析框架旨在通过因果探究干预模型对不同语言属性的内部表征，并测量模型在这些干预下与给定任务的基本事实因果模型的一致性，从而考察特定任务背景下的语言学习能力。我们还开发了一种使用基于梯度的对抗性攻击来执行因果探测干预的新方法，该方法可以针对比现有技术更广泛的属性和表示。最后，我们使用这些干预手段对CAMLE进行了个案研究，分析和比较了不同词汇推理任务的LLM能力，结果表明CAMPE可以用来解释和预测这些任务中的行为。



## **21. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



## **22. Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against Large Language Models**

留意您对世代的指导！探索针对大型语言模型的条件后门攻击 cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2404.14795v4) [paper-pdf](http://arxiv.org/pdf/2404.14795v4)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream backdoor attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of backdoor activation, we present a new poisoning paradigm against LLMs triggered by specifying generation conditions, which are commonly adopted strategies by users during model inference. The poisoned model performs normally for output under normal/other generation conditions, while becomes harmful for output under target generation conditions. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation conditions by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our attack can be generally divided into two types with different targets: Safety unalignment attack and Ability degradation attack. Our extensive experiments demonstrate that BrieFool is effective across safety domains and ability domains, achieving higher success rates than baseline methods, with 94.3 % on GPT-3.5-turbo

摘要: 针对大型语言模型(LLM)的主流后门攻击通常会在输入实例中设置固定的触发器，并为触发的查询设置特定的响应。然而，固定的触发设置(例如，不寻常的单词)可能很容易被人类检测到，从而限制了在现实世界场景中的有效性和实用性。为了增强后门激活的隐蔽性，我们提出了一种新的针对通过指定生成条件触发的LLM的中毒范例，这些策略是用户在模型推理中经常采用的策略。中毒模型在正常/其他发电条件下的出力表现正常，而在目标发电条件下的出力变得有害。为了实现这一目标，我们引入了一种高效的攻击框架BrieFool。它通过高效的指令采样和中毒数据生成来利用生成条件的特征，从而影响目标条件下的LLM的行为。我们的攻击一般可以分为两种类型，针对不同的目标：安全联盟攻击和能力退化攻击。我们的广泛实验表明，BrieFool跨安全域和能量域是有效的，获得了比基准方法更高的成功率，在GPT-3.5-Turbo上的成功率为94.3%



## **23. Large Language Models are Good Attackers: Efficient and Stealthy Textual Backdoor Attacks**

大型语言模型是好的攻击者：高效且隐秘的文本后门攻击 cs.CL

Under Review

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11587v1) [paper-pdf](http://arxiv.org/pdf/2408.11587v1)

**Authors**: Ziqiang Li, Yueqi Zeng, Pengfei Xia, Lei Liu, Zhangjie Fu, Bin Li

**Abstract**: With the burgeoning advancements in the field of natural language processing (NLP), the demand for training data has increased significantly. To save costs, it has become common for users and businesses to outsource the labor-intensive task of data collection to third-party entities. Unfortunately, recent research has unveiled the inherent risk associated with this practice, particularly in exposing NLP systems to potential backdoor attacks. Specifically, these attacks enable malicious control over the behavior of a trained model by poisoning a small portion of the training data. Unlike backdoor attacks in computer vision, textual backdoor attacks impose stringent requirements for attack stealthiness. However, existing attack methods meet significant trade-off between effectiveness and stealthiness, largely due to the high information entropy inherent in textual data. In this paper, we introduce the Efficient and Stealthy Textual backdoor attack method, EST-Bad, leveraging Large Language Models (LLMs). Our EST-Bad encompasses three core strategies: optimizing the inherent flaw of models as the trigger, stealthily injecting triggers with LLMs, and meticulously selecting the most impactful samples for backdoor injection. Through the integration of these techniques, EST-Bad demonstrates an efficient achievement of competitive attack performance while maintaining superior stealthiness compared to prior methods across various text classifier datasets.

摘要: 随着自然语言处理(NLP)领域的迅速发展，对训练数据的需求显著增加。为了节省成本，用户和企业将劳动密集型的数据收集任务外包给第三方实体已经变得很常见。不幸的是，最近的研究揭示了与这种做法相关的固有风险，特别是在将NLP系统暴露于潜在的后门攻击方面。具体地说，这些攻击通过对一小部分训练数据下毒，实现了对训练模型行为的恶意控制。与计算机视觉中的后门攻击不同，文本后门攻击对攻击的隐蔽性提出了严格的要求。然而，现有的攻击方法在有效性和隐蔽性之间达到了显著的权衡，这在很大程度上是由于文本数据固有的高信息熵。本文利用大型语言模型，介绍了一种高效、隐蔽的文本后门攻击方法EST-Bad。我们的EST-Bad包含三个核心策略：优化模型的固有缺陷作为触发器，悄悄地向触发器注入LLM，以及精心选择最有影响力的样本进行后门注入。通过这些技术的集成，EST-Bad展示了在与各种文本分类器数据集的现有方法相比保持优越的隐蔽性的同时，有效地实现了竞争性攻击性能。



## **24. SHIELD: Evaluation and Defense Strategies for Copyright Compliance in LLM Text Generation**

SHIELD：LLM文本生成中版权合规性的评估和防御策略 cs.CL

Work in progress

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.12975v2) [paper-pdf](http://arxiv.org/pdf/2406.12975v2)

**Authors**: Xiaoze Liu, Ting Sun, Tianyang Xu, Feijie Wu, Cunxiang Wang, Xiaoqian Wang, Jing Gao

**Abstract**: Large Language Models (LLMs) have transformed machine learning but raised significant legal concerns due to their potential to produce text that infringes on copyrights, resulting in several high-profile lawsuits. The legal landscape is struggling to keep pace with these rapid advancements, with ongoing debates about whether generated text might plagiarize copyrighted materials. Current LLMs may infringe on copyrights or overly restrict non-copyrighted texts, leading to these challenges: (i) the need for a comprehensive evaluation benchmark to assess copyright compliance from multiple aspects; (ii) evaluating robustness against safeguard bypassing attacks; and (iii) developing effective defense targeted against the generation of copyrighted text. To tackle these challenges, we introduce a curated dataset to evaluate methods, test attack strategies, and propose lightweight, real-time defense to prevent the generation of copyrighted text, ensuring the safe and lawful use of LLMs. Our experiments demonstrate that current LLMs frequently output copyrighted text, and that jailbreaking attacks can significantly increase the volume of copyrighted output. Our proposed defense mechanism significantly reduces the volume of copyrighted text generated by LLMs by effectively refusing malicious requests. Code is publicly available at https://github.com/xz-liu/SHIELD

摘要: 大型语言模型(LLM)改变了机器学习，但由于它们有可能产生侵犯版权的文本，因此引发了重大的法律担忧，导致了几起备受瞩目的诉讼。法律界正在努力跟上这些快速进步的步伐，关于生成的文本是否可能抄袭受版权保护的材料的争论仍在继续。当前的LLM可能会侵犯版权或过度限制非版权文本，从而导致以下挑战：(I)需要一个全面的评估基准来从多个方面评估版权合规性；(Ii)评估针对保护绕过攻击的稳健性；以及(Iii)针对版权文本的生成开发有效的防御措施。为了应对这些挑战，我们引入了一个经过精选的数据集来评估方法，测试攻击策略，并提出了轻量级、实时的防御措施来防止版权文本的生成，确保了LLMS的安全和合法使用。我们的实验表明，当前的LLM频繁地输出受版权保护的文本，越狱攻击可以显著增加受版权保护的输出量。我们提出的防御机制通过有效地拒绝恶意请求，大大减少了LLMS生成的受版权保护的文本的数量。代码可在https://github.com/xz-liu/SHIELD上公开获得



## **25. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

EEG-Defender：通过早期退出生成大型语言模型来抵御越狱 cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.

摘要: 大语言模型在各种应用中日益引起人们的关注。尽管如此，随着一些用户试图利用这些模型达到恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了减轻这种风险，人们提出了“对准”技术的概念。然而，最近的研究表明，这种对齐可以使用复杂的即时工程或敌对后缀来破坏，这是一种被称为“越狱”的技术。我们的研究从LLMS类似人类的生成过程中获得了线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示。利用这一发现，我们建议利用LLMS的早期变压器输出作为一种手段来检测恶意输入，并立即终止生成。基于这一想法，我们介绍了一种简单但重要的防御方法，称为用于LLMS的EEG-Defender。我们在三个模型上对十种越狱方法进行了全面的实验。我们的结果表明，EEG-Defender能够显著降低攻击成功率(ASR)，与现有SOTAS的50%相比，约为85%，而对LLMS的实用性和有效性的影响最小。



## **26. Medical MLLM is Vulnerable: Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models**

医学MLLM很脆弱：跨模式越狱和对医学多模式大型语言模型的不匹配攻击 cs.CR

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2405.20775v2) [paper-pdf](http://arxiv.org/pdf/2405.20775v2)

**Authors**: Xijie Huang, Xinyuan Wang, Hantao Zhang, Yinghao Zhu, Jiawen Xi, Jingkun An, Hao Wang, Hao Liang, Chengwei Pan

**Abstract**: Security concerns related to Large Language Models (LLMs) have been extensively explored, yet the safety implications for Multimodal Large Language Models (MLLMs), particularly in medical contexts (MedMLLMs), remain insufficiently studied. This paper delves into the underexplored security vulnerabilities of MedMLLMs, especially when deployed in clinical environments where the accuracy and relevance of question-and-answer interactions are critically tested against complex medical challenges. By combining existing clinical medical data with atypical natural phenomena, we define the mismatched malicious attack (2M-attack) and introduce its optimized version, known as the optimized mismatched malicious attack (O2M-attack or 2M-optimization). Using the voluminous 3MAD dataset that we construct, which covers a wide range of medical image modalities and harmful medical scenarios, we conduct a comprehensive analysis and propose the MCM optimization method, which significantly enhances the attack success rate on MedMLLMs. Evaluations with this dataset and attack methods, including white-box attacks on LLaVA-Med and transfer attacks (black-box) on four other SOTA models, indicate that even MedMLLMs designed with enhanced security features remain vulnerable to security breaches. Our work underscores the urgent need for a concerted effort to implement robust security measures and enhance the safety and efficacy of open-source MedMLLMs, particularly given the potential severity of jailbreak attacks and other malicious or clinically significant exploits in medical settings. Our code is available at https://github.com/dirtycomputer/O2M_attack.

摘要: 与大语言模型(LLM)相关的安全问题已经得到了广泛的研究，但多模式大语言模型(MLLM)的安全影响，特别是在医学背景下(MedMLLMS)的安全影响，仍然没有得到充分的研究。本文深入研究了MedMLLMS未被开发的安全漏洞，特别是当部署在临床环境中时，其中问答交互的准确性和相关性针对复杂的医疗挑战进行了严格的测试。结合已有的临床医学数据和非典型自然现象，我们定义了失配恶意攻击(2M-Attack)，并介绍了其优化版本，称为优化失配恶意攻击(O2M-Attack或2M-Optimation)。利用我们构建的涵盖多种医学图像模式和有害医疗场景的海量3MAD数据集，进行了全面的分析，并提出了MCM优化方法，显著提高了对MedMLLms的攻击成功率。使用该数据集和攻击方法进行的评估，包括对LLaVA-Med的白盒攻击和对其他四个SOTA型号的传输攻击(黑盒)，表明即使是设计了增强安全功能的MedMLLM也仍然容易受到安全漏洞的攻击。我们的工作强调了迫切需要共同努力，实施强有力的安全措施，提高开源MedMLLMS的安全性和有效性，特别是考虑到越狱攻击和医疗环境中其他恶意或具有临床意义的利用的潜在严重性。我们的代码可以在https://github.com/dirtycomputer/O2M_attack.上找到



## **27. Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Neural Carrier Articles**

将你的恶意目标隐藏在良性叙述中：通过神经载体文章越狱大型语言模型 cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11182v1) [paper-pdf](http://arxiv.org/pdf/2408.11182v1)

**Authors**: Zhilong Wang, Haizhou Wang, Nanqing Luo, Lan Zhang, Xiaoyan Sun, Yebo Cao, Peng Liu

**Abstract**: Jailbreak attacks on Language Model Models (LLMs) entail crafting prompts aimed at exploiting the models to generate malicious content. This paper proposes a new type of jailbreak attacks which shift the attention of the LLM by inserting a prohibited query into a carrier article. The proposed attack leverage the knowledge graph and a composer LLM to automatically generating a carrier article that is similar to the topic of the prohibited query but does not violate LLM's safeguards. By inserting the malicious query to the carrier article, the assembled attack payload can successfully jailbreak LLM. To evaluate the effectiveness of our method, we leverage 4 popular categories of ``harmful behaviors'' adopted by related researches to attack 6 popular LLMs. Our experiment results show that the proposed attacking method can successfully jailbreak all the target LLMs which high success rate, except for Claude-3.

摘要: 对语言模型模型（LLM）的越狱攻击需要精心设计旨在利用模型生成恶意内容的提示。本文提出了一种新型越狱攻击，通过在载体文章中插入被禁止的查询来转移LLM的注意力。拟议的攻击利用知识图和作曲家LLM来自动生成与禁止查询的主题相似但不违反LLM的保障措施的载体文章。通过将恶意查询插入到载体文章中，组装的攻击有效负载可以成功越狱LLM。为了评估我们方法的有效性，我们利用相关研究采用的4种流行类别“有害行为”来攻击6种流行的LLM。实验结果表明，所提出的攻击方法可以成功越狱除Claude-3外的所有成功率较高的目标LLM。



## **28. Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks**

羊皮中的假新闻：针对LLM授权的风格攻击的强大假新闻检测 cs.CL

Accepted to KDD 2024 (Research Track)

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2310.10830v2) [paper-pdf](http://arxiv.org/pdf/2310.10830v2)

**Authors**: Jiaying Wu, Jiafeng Guo, Bryan Hooi

**Abstract**: It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential intepretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.

摘要: 人们普遍认为，假新闻和真实新闻表现出截然不同的写作风格，例如使用耸人听闻的语言与客观语言。但是，我们强调，与样式相关的功能也可以用于基于样式的攻击。值得注意的是，强大的大型语言模型(LLM)的出现使恶意攻击者能够快速、经济、大规模地模仿可信新闻来源的风格。我们的分析显示，LLM伪装的假新闻内容显著削弱了最先进的基于文本的检测器的有效性(F1分数下降了38%)，这意味着对文体变化的严重脆弱性。为了解决这个问题，我们引入了SheepDog，这是一个风格稳健的假新闻检测器，在确定新闻真实性时，它将内容优先于风格。Sheepog通过以下方式实现这种弹性：(1)LLM授权的新闻重组，通过定制文章以匹配不同的风格，将风格多样性注入训练过程；(2)风格不可知的培训方案，确保在风格多样化的重组中一致的准确性预测；以及(3)以内容为中心的准确性归因，从LLMS中提取以内容为中心的指导方针，以揭穿假新闻，提供补充线索和潜在的可理解性，以帮助准确性预测。在三个真实世界的基准上进行的广泛实验表明，牧羊犬的风格、健壮性和对各种主干的适应性。



## **29. While GitHub Copilot Excels at Coding, Does It Ensure Responsible Output?**

虽然GitHub Copilot擅长编码，但它能否确保负责任的输出？ cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11006v1) [paper-pdf](http://arxiv.org/pdf/2408.11006v1)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **30. Towards Efficient Formal Verification of Spiking Neural Network**

尖峰神经网络的有效形式验证 cs.AI

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10900v1) [paper-pdf](http://arxiv.org/pdf/2408.10900v1)

**Authors**: Baekryun Seong, Jieung Kim, Sang-Ki Ko

**Abstract**: Recently, AI research has primarily focused on large language models (LLMs), and increasing accuracy often involves scaling up and consuming more power. The power consumption of AI has become a significant societal issue; in this context, spiking neural networks (SNNs) offer a promising solution. SNNs operate event-driven, like the human brain, and compress information temporally. These characteristics allow SNNs to significantly reduce power consumption compared to perceptron-based artificial neural networks (ANNs), highlighting them as a next-generation neural network technology. However, societal concerns regarding AI go beyond power consumption, with the reliability of AI models being a global issue. For instance, adversarial attacks on AI models are a well-studied problem in the context of traditional neural networks. Despite their importance, the stability and property verification of SNNs remains in the early stages of research. Most SNN verification methods are time-consuming and barely scalable, making practical applications challenging. In this paper, we introduce temporal encoding to achieve practical performance in verifying the adversarial robustness of SNNs. We conduct a theoretical analysis of this approach and demonstrate its success in verifying SNNs at previously unmanageable scales. Our contribution advances SNN verification to a practical level, facilitating the safer application of SNNs.

摘要: 最近，人工智能的研究主要集中在大型语言模型(LLM)上，而提高精确度往往需要扩大规模和消耗更多功率。人工智能的能耗已经成为一个重要的社会问题；在这种背景下，尖峰神经网络(SNN)提供了一个有前途的解决方案。SNN像人脑一样，以事件驱动的方式运行，并在时间上压缩信息。与基于感知器的人工神经网络(ANN)相比，这些特性使SNN能够显著降低功耗，突出了它们作为下一代神经网络技术的重要性。然而，社会对人工智能的担忧不仅仅是电力消耗，人工智能模型的可靠性是一个全球问题。例如，在传统神经网络的背景下，对人工智能模型的敌意攻击是一个研究得很好的问题。尽管它们很重要，但SNN的稳定性和性质验证仍处于研究的早期阶段。大多数SNN验证方法都很耗时且几乎不可扩展，这给实际应用带来了挑战。在本文中，我们引入时间编码以达到在验证SNN的对抗健壮性方面的实际性能。我们对这种方法进行了理论分析，并证明了它在以前无法管理的规模上验证SNN的成功。我们的贡献将SNN验证提升到了一个实用的水平，促进了SNN的更安全应用。



## **31. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10738v1) [paper-pdf](http://arxiv.org/pdf/2408.10738v1)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also encounter notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the top k relevant items from offline knowledge bases, utilizing all available information from a webpage, including logos, HTML, and URLs. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也遇到了显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们提出了一个多通道信息检索框架，旨在利用网页中的所有可用信息，包括徽标、HTML和URL，从离线知识库中提取前k个相关条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **32. MEGen: Generative Backdoor in Large Language Models via Model Editing**

MEGen：通过模型编辑在大型语言模型中实现生成后门 cs.CL

Working in progress

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10722v1) [paper-pdf](http://arxiv.org/pdf/2408.10722v1)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities. Their powerful generative abilities enable flexible responses based on various queries or instructions. Emerging as widely adopted generalists for diverse tasks, LLMs are still vulnerable to backdoors. This paper proposes an editing-based generative backdoor, named MEGen, aiming to create a customized backdoor for NLP tasks with the least side effects. In our approach, we first leverage a language model to insert a trigger selected on fixed metrics into the input, then design a pipeline of model editing to directly embed a backdoor into an LLM. By adjusting a small set of local parameters with a mini-batch of samples, MEGen significantly enhances time efficiency and achieves high robustness. Experimental results indicate that our backdoor attack strategy achieves a high attack success rate on poison data while maintaining the model's performance on clean data. Notably, the backdoored model, when triggered, can freely output pre-set dangerous information while successfully completing downstream tasks. This suggests that future LLM applications could be guided to deliver certain dangerous information, thus altering the LLM's generative style. We believe this approach provides insights for future LLM applications and the execution of backdoor attacks on conversational AI systems.

摘要: 大型语言模型(LLM)已经显示出非凡的能力。它们强大的生成能力使人们能够根据各种查询或指令做出灵活的反应。作为在不同任务中被广泛采用的多面手，LLM仍很容易受到后门的攻击。本文提出了一种基于编辑的生成性后门Megen，旨在为NLP任务创建一个具有最小副作用的定制后门。在我们的方法中，我们首先利用语言模型将根据固定指标选择的触发器插入到输入中，然后设计一个模型编辑管道来直接将后门嵌入到LLM中。通过用一小批样本调整一小组局部参数，Megen显著提高了时间效率并实现了高稳健性。实验结果表明，我们的后门攻击策略在保持模型对干净数据的性能的同时，对有毒数据取得了较高的攻击成功率。值得注意的是，后置模型在被触发时，可以在成功完成下游任务的同时自由输出预设的危险信息。这表明，未来的LLM应用程序可能会被引导来传递某些危险的信息，从而改变LLM的生成风格。我们相信，这种方法为未来的LLM应用程序和对对话式人工智能系统执行后门攻击提供了见解。



## **33. Ferret: Faster and Effective Automated Red Teaming with Reward-Based Scoring Technique**

Ferret：更快、更有效的自动化红色团队，采用基于奖励的评分技术 cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10701v1) [paper-pdf](http://arxiv.org/pdf/2408.10701v1)

**Authors**: Tej Deep Pala, Vernon Y. H. Toh, Rishabh Bhardwaj, Soujanya Poria

**Abstract**: In today's era, where large language models (LLMs) are integrated into numerous real-world applications, ensuring their safety and robustness is crucial for responsible AI usage. Automated red-teaming methods play a key role in this process by generating adversarial attacks to identify and mitigate potential vulnerabilities in these models. However, existing methods often struggle with slow performance, limited categorical diversity, and high resource demands. While Rainbow Teaming, a recent approach, addresses the diversity challenge by framing adversarial prompt generation as a quality-diversity search, it remains slow and requires a large fine-tuned mutator for optimal performance. To overcome these limitations, we propose Ferret, a novel approach that builds upon Rainbow Teaming by generating multiple adversarial prompt mutations per iteration and using a scoring function to rank and select the most effective adversarial prompt. We explore various scoring functions, including reward models, Llama Guard, and LLM-as-a-judge, to rank adversarial mutations based on their potential harm to improve the efficiency of the search for harmful mutations. Our results demonstrate that Ferret, utilizing a reward model as a scoring function, improves the overall attack success rate (ASR) to 95%, which is 46% higher than Rainbow Teaming. Additionally, Ferret reduces the time needed to achieve a 90% ASR by 15.2% compared to the baseline and generates adversarial prompts that are transferable i.e. effective on other LLMs of larger size. Our codes are available at https://github.com/declare-lab/ferret.

摘要: 在当今时代，大型语言模型(LLM)被集成到许多现实世界的应用程序中，确保它们的安全性和健壮性对于负责任的人工智能使用至关重要。自动红团队方法通过生成对抗性攻击来识别和缓解这些模型中的潜在漏洞，从而在这一过程中发挥关键作用。然而，现有的方法往往在性能缓慢、分类多样性有限和资源需求高的情况下苦苦挣扎。虽然彩虹组合是最近的一种方法，通过将敌意提示生成框定为一种质量多样性搜索来解决多样性挑战，但它仍然很慢，需要一个大型微调赋值器来实现最佳性能。为了克服这些局限性，我们提出了一种新的方法--FERRET，它建立在彩虹分组的基础上，通过每次迭代产生多个对抗性提示突变，并使用评分函数来对最有效的对抗性提示进行排序和选择。我们探索了各种评分函数，包括奖励模型、骆驼警卫和LLM作为法官，根据潜在的危害对对手突变进行排名，以提高有害突变的搜索效率。我们的结果表明，利用奖励模型作为得分函数的雪貂，将总体攻击成功率(ASR)提高到95%，比彩虹组合高出46%。此外，与基线相比，雪貂将达到90%的ASR所需的时间减少了15.2%，并生成可转移的对抗性提示，即对其他较大规模的LLM有效。我们的代码可在https://github.com/declare-lab/ferret.上获得



## **34. Promoting Equality in Large Language Models: Identifying and Mitigating the Implicit Bias based on Bayesian Theory**

促进大型语言模型中的平等：基于Bayesian理论识别和缓解隐性偏见 cs.CL

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10608v1) [paper-pdf](http://arxiv.org/pdf/2408.10608v1)

**Authors**: Yongxin Deng, Xihe Qiu, Xiaoyu Tan, Jing Pan, Chen Jue, Zhijun Fang, Yinghui Xu, Wei Chu, Yuan Qi

**Abstract**: Large language models (LLMs) are trained on extensive text corpora, which inevitably include biased information. Although techniques such as Affective Alignment can mitigate some negative impacts of these biases, existing prompt-based attack methods can still extract these biases from the model's weights. Moreover, these biases frequently appear subtly when LLMs are prompted to perform identical tasks across different demographic groups, thereby camouflaging their presence. To address this issue, we have formally defined the implicit bias problem and developed an innovative framework for bias removal based on Bayesian theory, Bayesian-Theory based Bias Removal (BTBR). BTBR employs likelihood ratio screening to pinpoint data entries within publicly accessible biased datasets that represent biases inadvertently incorporated during the LLM training phase. It then automatically constructs relevant knowledge triples and expunges bias information from LLMs using model editing techniques. Through extensive experimentation, we have confirmed the presence of the implicit bias problem in LLMs and demonstrated the effectiveness of our BTBR approach.

摘要: 大型语言模型(LLM)是在广泛的文本语料库上训练的，其中不可避免地包含有偏见的信息。虽然情感对齐等技术可以缓解这些偏差的一些负面影响，但现有的基于提示的攻击方法仍然可以从模型的权重中提取这些偏差。此外，当LLM被提示在不同的人口群体中执行相同的任务时，这些偏见经常微妙地出现，从而掩盖了他们的存在。为了解决这个问题，我们正式定义了隐含偏差问题，并在贝叶斯理论的基础上提出了一种新的去偏框架--基于贝叶斯理论的去偏方法(BTBR)。BTBR使用似然比筛选来精确定位可公开访问的有偏数据集中的数据条目，这些数据表示在LLM训练阶段无意中并入的偏差。然后利用模型编辑技术自动构建相关知识三元组，并从低似然模型中剔除偏差信息。通过大量的实验，我们证实了LLMS中隐含偏差问题的存在，并证明了我们的BTBR方法的有效性。



## **35. PromptBench: A Unified Library for Evaluation of Large Language Models**

EntBench：大型语言模型评估的统一库 cs.AI

Accepted by Journal of Machine Learning Research (JMLR); code:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2312.07910v3) [paper-pdf](http://arxiv.org/pdf/2312.07910v3)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型（LLM）的评估对于评估其性能和降低潜在的安全风险至关重要。本文中，我们介绍了EmotiBench，这是一个评估LLM的统一图书馆。它由研究人员易于使用和扩展的几个关键组件组成：即时构建、即时工程、数据集和模型加载、对抗即时攻击、动态评估协议和分析工具。EntBench旨在成为一个开放、通用和灵活的代码库，用于研究目的，可以促进创建新基准、部署下游应用程序和设计新评估协议的原始研究。该代码可在https://github.com/microsoft/promptbench上获取，并将持续支持。



## **36. Development of an AI Anti-Bullying System Using Large Language Model Key Topic Detection**

基于大语言模型关键话题检测的人工智能反欺凌系统开发 cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10417v1) [paper-pdf](http://arxiv.org/pdf/2408.10417v1)

**Authors**: Matthew Tassava, Cameron Kolodjski, Jordan Milbrath, Adorah Bishop, Nathan Flanders, Robbie Fetsch, Danielle Hanson, Jeremy Straub

**Abstract**: This paper presents and evaluates work on the development of an artificial intelligence (AI) anti-bullying system. The system is designed to identify coordinated bullying attacks via social media and other mechanisms, characterize them and propose remediation and response activities to them. In particular, a large language model (LLM) is used to populate an enhanced expert system-based network model of a bullying attack. This facilitates analysis and remediation activity - such as generating report messages to social media companies - determination. The system is described and the efficacy of the LLM for populating the model is analyzed herein.

摘要: 本文介绍并评估了人工智能（AI）反欺凌系统的开发工作。该系统旨在识别通过社交媒体和其他机制的协调欺凌攻击，对其进行特征描述并提出补救和响应活动。特别是，使用大型语言模型（LLM）来填充欺凌攻击的增强型基于专家系统的网络模型。这有助于分析和补救活动（例如向社交媒体公司生成报告消息）的确定。本文描述了该系统，并分析了LLM填充模型的功效。



## **37. A Disguised Wolf Is More Harmful Than a Toothless Tiger: Adaptive Malicious Code Injection Backdoor Attack Leveraging User Behavior as Triggers**

伪装的狼比无牙的老虎更有害：利用用户行为作为触发器的自适应恶意代码注入后门攻击 cs.AI

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10334v1) [paper-pdf](http://arxiv.org/pdf/2408.10334v1)

**Authors**: Shangxi Wu, Jitao Sang

**Abstract**: In recent years, large language models (LLMs) have made significant progress in the field of code generation. However, as more and more users rely on these models for software development, the security risks associated with code generation models have become increasingly significant. Studies have shown that traditional deep learning robustness issues also negatively impact the field of code generation. In this paper, we first present the game-theoretic model that focuses on security issues in code generation scenarios. This framework outlines possible scenarios and patterns where attackers could spread malicious code models to create security threats. We also pointed out for the first time that the attackers can use backdoor attacks to dynamically adjust the timing of malicious code injection, which will release varying degrees of malicious code depending on the skill level of the user. Through extensive experiments on leading code generation models, we validate our proposed game-theoretic model and highlight the significant threats that these new attack scenarios pose to the safe use of code models.

摘要: 近年来，大型语言模型(LLM)在代码生成领域取得了重大进展。然而，随着越来越多的用户依赖这些模型进行软件开发，与代码生成模型相关的安全风险也变得越来越严重。研究表明，传统的深度学习健壮性问题也对代码生成领域产生了负面影响。在本文中，我们首先提出了关注代码生成场景中的安全问题的博弈论模型。此框架概述了攻击者传播恶意代码模型以制造安全威胁的可能场景和模式。我们还首次指出，攻击者可以利用后门攻击来动态调整恶意代码注入的时间，这将根据用户的技能水平释放不同程度的恶意代码。通过在领先的代码生成模型上的大量实验，我们验证了我们提出的博弈论模型，并强调了这些新的攻击场景对代码模型的安全使用构成的重大威胁。



## **38. Topic-Based Watermarks for LLM-Generated Text**

LLM生成文本的基于主题的水印 cs.CR

Results for proposed scheme, additional/removal of content (figures  and equations), 12 pages

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2404.02138v3) [paper-pdf](http://arxiv.org/pdf/2404.02138v3)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of text generated by large language models (LLMs) from human-generated text poses significant challenges. Watermarking algorithms are potential solutions by embedding detectable signatures within LLM-generated outputs. However, current watermarking schemes lack robustness to a range of attacks such as text substitution or manipulation, undermining their reliability. This paper proposes a novel topic-based watermarking algorithm for LLMs, designed to enhance the robustness of watermarking in LLMs. Our approach leverages the topics extracted from input prompts or outputs of non-watermarked LLMs in the generation process of watermarked text. We dynamically utilize token lists on identified topics and adjust token sampling weights accordingly. By using these topic-specific token biases, we embed a topic-sensitive watermarking into the generated text. We outline the theoretical framework of our topic-based watermarking algorithm and discuss its potential advantages in various scenarios. Additionally, we explore a comprehensive range of attacks against watermarking algorithms, including discrete alterations, paraphrasing, and tokenizations. We demonstrate that our proposed watermarking scheme classifies various watermarked text topics with 99.99% confidence and outperforms existing algorithms in terms of z-score robustness and the feasibility of modeling text degradation by potential attackers, while considering the trade-offs between the benefits and losses of watermarking LLM-generated text.

摘要: 大型语言模型(LLM)生成的文本与人类生成的文本无法区分，这带来了巨大的挑战。通过在LLM生成的输出中嵌入可检测的签名，水印算法是潜在的解决方案。然而，当前的水印方案缺乏对文本替换或篡改等一系列攻击的稳健性，从而破坏了它们的可靠性。提出了一种新的基于主题的LLMS水印算法，旨在增强LLMS中水印的稳健性。我们的方法在水印文本的生成过程中利用了从非水印LLMS的输入提示或输出中提取的主题。我们在识别的主题上动态地利用令牌列表，并相应地调整令牌抽样权重。通过使用这些特定于主题的标记偏差，我们在生成的文本中嵌入了主题敏感的水印。我们概述了我们的基于主题的水印算法的理论框架，并讨论了它在各种场景下的潜在优势。此外，我们还探讨了针对水印算法的广泛攻击，包括离散更改、释义和标记化。我们证明了我们提出的水印方案以99.99%的置信度对不同的水印文本主题进行分类，并且在z-Score稳健性和潜在攻击者对文本退化建模的可行性方面优于现有算法，同时考虑了在LLM生成的文本中添加水印的利弊权衡。



## **39. Privacy Checklist: Privacy Violation Detection Grounding on Contextual Integrity Theory**

隐私检查表：基于上下文完整性理论的隐私侵犯检测 cs.CL

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.10053v1) [paper-pdf](http://arxiv.org/pdf/2408.10053v1)

**Authors**: Haoran Li, Wei Fan, Yulin Chen, Jiayang Cheng, Tianshu Chu, Xuebing Zhou, Peizhao Hu, Yangqiu Song

**Abstract**: Privacy research has attracted wide attention as individuals worry that their private data can be easily leaked during interactions with smart devices, social platforms, and AI applications. Computer science researchers, on the other hand, commonly study privacy issues through privacy attacks and defenses on segmented fields. Privacy research is conducted on various sub-fields, including Computer Vision (CV), Natural Language Processing (NLP), and Computer Networks. Within each field, privacy has its own formulation. Though pioneering works on attacks and defenses reveal sensitive privacy issues, they are narrowly trapped and cannot fully cover people's actual privacy concerns. Consequently, the research on general and human-centric privacy research remains rather unexplored. In this paper, we formulate the privacy issue as a reasoning problem rather than simple pattern matching. We ground on the Contextual Integrity (CI) theory which posits that people's perceptions of privacy are highly correlated with the corresponding social context. Based on such an assumption, we develop the first comprehensive checklist that covers social identities, private attributes, and existing privacy regulations. Unlike prior works on CI that either cover limited expert annotated norms or model incomplete social context, our proposed privacy checklist uses the whole Health Insurance Portability and Accountability Act of 1996 (HIPAA) as an example, to show that we can resort to large language models (LLMs) to completely cover the HIPAA's regulations. Additionally, our checklist also gathers expert annotations across multiple ontologies to determine private information including but not limited to personally identifiable information (PII). We use our preliminary results on the HIPAA to shed light on future context-centric privacy research to cover more privacy regulations, social norms and standards.

摘要: 隐私研究吸引了广泛的关注，因为个人担心他们的私人数据在与智能设备、社交平台和人工智能应用程序交互时很容易被泄露。另一方面，计算机科学研究人员通常通过对分割的领域进行隐私攻击和防御来研究隐私问题。隐私研究在不同的子领域进行，包括计算机视觉(CV)、自然语言处理(NLP)和计算机网络。在每个领域，隐私都有自己的表述。尽管攻击和防御方面的开创性作品揭示了敏感的隐私问题，但它们被狭隘地困住了，不能完全覆盖人们对隐私的实际担忧。因此，关于一般隐私研究和以人为中心的隐私研究仍然是相当未被探索的。在本文中，我们将隐私问题描述为一个推理问题，而不是简单的模式匹配。我们基于语境完整性(CI)理论，该理论认为人们对隐私的感知与相应的社会语境高度相关。基于这样的假设，我们开发了第一个全面的清单，其中包括社会身份、私人属性和现有的隐私法规。与以往关于CI的工作要么涵盖有限的专家注释规范，要么涵盖不完整的社会背景，我们提出的隐私检查表以1996年的整个健康保险携带和责任法案(HIPAA)为例，表明我们可以求助于大型语言模型(LLM)来完全覆盖HIPAA的规定。此外，我们的检查表还收集了跨多个本体的专家注释，以确定私人信息，包括但不限于个人身份信息(PII)。我们使用我们在HIPAA上的初步结果来阐明未来以上下文为中心的隐私研究，以涵盖更多的隐私法规、社会规范和标准。



## **40. Transferring Backdoors between Large Language Models by Knowledge Distillation**

通过知识蒸馏在大型语言模型之间转移后门 cs.CR

13 pages, 16 figures, 5 tables

**SubmitDate**: 2024-08-19    [abs](http://arxiv.org/abs/2408.09878v1) [paper-pdf](http://arxiv.org/pdf/2408.09878v1)

**Authors**: Pengzhou Cheng, Zongru Wu, Tianjie Ju, Wei Du, Zhuosheng Zhang Gongshen Liu

**Abstract**: Backdoor Attacks have been a serious vulnerability against Large Language Models (LLMs). However, previous methods only reveal such risk in specific models, or present tasks transferability after attacking the pre-trained phase. So, how risky is the model transferability of a backdoor attack? In this paper, we focus on whether existing mini-LLMs may be unconsciously instructed in backdoor knowledge by poisoned teacher LLMs through knowledge distillation (KD). Specifically, we propose ATBA, an adaptive transferable backdoor attack, which can effectively distill the backdoor of teacher LLMs into small models when only executing clean-tuning. We first propose the Target Trigger Generation (TTG) module that filters out a set of indicative trigger candidates from the token list based on cosine similarity distribution. Then, we exploit a shadow model to imitate the distilling process and introduce an Adaptive Trigger Optimization (ATO) module to realize a gradient-based greedy feedback to search optimal triggers. Extensive experiments show that ATBA generates not only positive guidance for student models but also implicitly transfers backdoor knowledge. Our attack is robust and stealthy, with over 80% backdoor transferability, and hopes the attention of security.

摘要: 后门攻击一直是大型语言模型(LLM)的一个严重漏洞。然而，以前的方法只在特定的模型中发现这种风险，或者在攻击预训练阶段之后呈现任务的可转移性。那么，后门攻击的模型可转移性有多大风险呢？在本文中，我们关注现有的微型LLM是否可能被中毒的教师LLMS通过知识蒸馏(KD)无意识地传授到后门知识。具体地说，我们提出了一种自适应可转移后门攻击ATBA，它可以在只执行干净调优的情况下有效地将教师LLM的后门提取成小模型。我们首先提出了目标触发器生成(TTG)模块，该模块根据余弦相似度分布从令牌列表中过滤出一组指示性触发器候选。然后，我们利用影子模型来模拟提取过程，并引入自适应触发优化(ATO)模块来实现基于梯度的贪婪反馈来搜索最优触发。大量的实验表明，ATBA不仅为学生模型生成了积极的指导，而且还隐式地传递了后门知识。我们的攻击具有健壮性和隐蔽性，具有80%以上的后门可转移性，希望引起安全方面的关注。



## **41. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

Bergeron：通过基于意识的一致框架打击敌对攻击 cs.CR

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2312.00029v3) [paper-pdf](http://arxiv.org/pdf/2312.00029v3)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. Such vulnerabilities can lead to LLMs being manipulated into generating hazardous content: from instructions for creating dangerous materials to inciting violence or endorsing unethical behaviors. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM acting as a guardian to the primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis reviews that by using Bergeron to complement models with existing alignment training, we can significantly improve the robustness and safety of multiple, commonly used commercial and open-source LLMs. Specifically, we found that models integrated with Bergeron are, on average, nearly seven times more resistant to attacks compared to models without such support.

摘要: 自从最近引入了功能越来越强大的大型语言模型(LLM)以来，对人工智能对齐的研究有了很大的增长。不幸的是，现代的校准方法仍然不能完全防止模型受到故意攻击时的有害反应。这些漏洞可能导致LLMS被操纵来生成危险内容：从创建危险材料的说明到煽动暴力或支持不道德行为。为了帮助缓解这个问题，我们引入了Bergeron：一个旨在提高LLM抵御攻击的健壮性的框架，而不需要任何额外的参数微调。Bergeron被组织成两级；辅助LLM充当主要LLM的监护人。此框架可以更好地保护主要模型免受来袭攻击，同时监控其输出中是否有任何有害内容。经验分析认为，通过使用Bergeron来补充模型与现有的比对训练，我们可以显著提高多个常用的商业和开源LLM的稳健性和安全性。具体地说，我们发现，与没有这种支持的型号相比，集成了Bergeron的型号平均抵抗攻击的能力要高出近7倍。



## **42. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

解药：微调后大型语言模型的安全调整，防止有害的微调 cs.AI

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09600v1) [paper-pdf](http://arxiv.org/pdf/2408.09600v1)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks \cite{qi2023fine}-- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. Existing mitigation strategies include alignment stage solutions \cite{huang2024vaccine, rosati2024representation} and fine-tuning stage solutions \cite{huang2024lazy,mukhoti2023fine}. However, our evaluation shows that both categories of defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense, which however, is necessary to guarantee finetune performance. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks.

摘要: 安全对齐的大型语言模型(LLM)容易受到有害的微调攻击--在微调数据集中混合一些有害数据就会破坏LLMS的安全对齐。现有的缓解策略包括对齐阶段解决方案\cite{huang2024疫苗，rosati2024代表}和微调阶段解决方案\cite{huang2024 lazy，mukhoti2023 finy}。然而，我们的评估表明，这两类防御都失败了[当选择了一些特定的训练超参数}--在微调阶段，较大的学习速率或大量的训练周期很容易使防御失效，但这是保证精调性能所必需的。为此，我们提出了解毒剂，这是一种后微调阶段的解决方案，它仍然保持在文本bf{与微调阶段的训练超参数无关}}。解毒剂依靠的理念是，通过删除有害参数，可以从有害行为中恢复有害模型，而无论这些有害参数在微调阶段是如何形成的。本着这一理念，我们引入了有害微调后的一次修剪阶段，以去除导致有害内容生成的有害权重。尽管解毒剂简单得令人尴尬，但实验结果表明，解毒剂可以降低有害分数，同时保持下游任务的准确性。



## **43. Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks**

描述和评估LLM针对越狱攻击的可靠性 cs.CL

**SubmitDate**: 2024-08-18    [abs](http://arxiv.org/abs/2408.09326v1) [paper-pdf](http://arxiv.org/pdf/2408.09326v1)

**Authors**: Kexin Chen, Yi Liu, Dongxia Wang, Jiaying Chen, Wenhai Wang

**Abstract**: Large Language Models (LLMs) have increasingly become pivotal in content generation with notable societal impact. These models hold the potential to generate content that could be deemed harmful.Efforts to mitigate this risk include implementing safeguards to ensure LLMs adhere to social ethics.However, despite such measures, the phenomenon of "jailbreaking" -- where carefully crafted prompts elicit harmful responses from models -- persists as a significant challenge. Recognizing the continuous threat posed by jailbreaking tactics and their repercussions for the trustworthy use of LLMs, a rigorous assessment of the models' robustness against such attacks is essential. This study introduces an comprehensive evaluation framework and conducts an large-scale empirical experiment to address this need. We concentrate on 10 cutting-edge jailbreak strategies across three categories, 1525 questions from 61 specific harmful categories, and 13 popular LLMs. We adopt multi-dimensional metrics such as Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, and Grammatical Errors to thoroughly assess the LLMs' outputs under jailbreak. By normalizing and aggregating these metrics, we present a detailed reliability score for different LLMs, coupled with strategic recommendations to reduce their susceptibility to such vulnerabilities. Additionally, we explore the relationships among the models, attack strategies, and types of harmful content, as well as the correlations between the evaluation metrics, which proves the validity of our multifaceted evaluation framework. Our extensive experimental results demonstrate a lack of resilience among all tested LLMs against certain strategies, and highlight the need to concentrate on the reliability facets of LLMs. We believe our study can provide valuable insights into enhancing the security evaluation of LLMs against jailbreak within the domain.

摘要: 大型语言模型(LLM)已日益成为内容生成的关键，具有显著的社会影响。这些模式有可能产生可能被认为有害的内容。缓解这种风险的方法包括实施保障措施，以确保LLMS遵守社会道德。然而，尽管采取了这些措施，“越狱”现象--精心制作的提示会招致模特的有害回应--仍然是一个重大挑战。认识到越狱战术构成的持续威胁及其对可信地使用LLM的影响，严格评估这些模型对此类攻击的稳健性是至关重要的。为了满足这一需求，本研究引入了一个综合评价框架，并进行了大规模的实证实验。我们集中在三个类别的10个尖端越狱策略，来自61个特定有害类别的1525个问题，以及13个流行的LLM。我们采用攻击成功率(ASR)、毒性分数、流畅度、令牌长度和语法错误等多维度量来彻底评估越狱情况下LLMS的输出。通过标准化和聚合这些指标，我们为不同的LLM提供了详细的可靠性分数，并提供了降低它们对此类漏洞的易感性的战略建议。此外，我们还探讨了模型、攻击策略和有害内容类型之间的关系，以及评估指标之间的相关性，从而证明了我们的多方面评估框架的有效性。我们广泛的实验结果表明，在所有测试的LLM中，对某些策略缺乏弹性，并强调了需要专注于LLM的可靠性方面。我们相信，我们的研究可以为加强LLMS在域内抗越狱的安全性评估提供有价值的见解。



## **44. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2024-08-17    [abs](http://arxiv.org/abs/2408.09093v1) [paper-pdf](http://arxiv.org/pdf/2408.09093v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多通道大型语言模型(MLLM)在各种多通道任务中表现出令人印象深刻的性能。另一方面，附加图像模式的集成可能允许恶意用户在图像中注入有害内容以越狱。与基于文本的LLMS不同，在LLMS中，攻击者需要使用特定的算法选择离散的令牌来隐藏其恶意意图，而图像信号的连续性为攻击者提供了直接注入有害意图的机会。在这项工作中，我们提出了一种简单而有效的越狱防御机制--$\extbf{bathe}$($\extbf{ba}$ck door$\extbf{T}$rigger S$\extbf{h}$i$\extbf{e}$ld)。我们的工作是基于生成式语言模型对越狱后门攻击和虚拟提示后门攻击的最新研究。越狱后门攻击使用有害指令和手动创建的字符串作为触发器，使后门模型生成被禁止的响应。我们假设有害指令可以作为触发器，如果我们将拒绝响应设置为触发响应，那么反向模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一点，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为‘’楔形‘’。我们的综合实验表明，BAIT有效地缓解了各种类型的越狱攻击，并且能够自适应地防御看不见的攻击，对MLLMS的性能影响最小。



## **45. Can Editing LLMs Inject Harm?**

编辑LLM会造成伤害吗？ cs.CL

The first two authors contributed equally. 9 pages for main paper, 36  pages including appendix. The code, results, dataset for this paper and more  resources are on the project website: https://llm-editing.github.io

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.20224v3) [paper-pdf](http://arxiv.org/pdf/2407.20224v3)

**Authors**: Canyu Chen, Baixiang Huang, Zekun Li, Zhaorun Chen, Shiyang Lai, Xiongxiao Xu, Jia-Chen Gu, Jindong Gu, Huaxiu Yao, Chaowei Xiao, Xifeng Yan, William Yang Wang, Philip Torr, Dawn Song, Kai Shu

**Abstract**: Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.

摘要: 在大型语言模型中，知识编辑被越来越多地用于纠正错误或过时的知识。与此同时，一个关键但未被探讨的问题是：知识编辑能否被用来向低收入国家注入危害？在本文中，我们将知识编辑重新定义为一种新的安全威胁，即编辑攻击，并使用新构建的数据集EditAttack进行了系统的研究。具体地说，我们重点研究了编辑攻击的两个典型的安全风险，包括错误信息注入和偏见注入。对于错误信息注入的风险，我们首先将其分为常识性错误信息注入和长尾错误信息注入。然后，我们发现编辑攻击可以将这两种类型的错误信息注入到LLMS中，其中常识性错误信息注入的有效性尤其高。对于偏向注入的风险，我们发现，偏向句不仅可以被高效地注入到LLMS中，而且一次偏向句注入会导致LLMS的总体输出出现偏向增加，甚至与注入的句子高度无关，这对LLMS的整体公平性造成了灾难性的影响。然后，我们进一步说明了编辑攻击的高度隐蔽性，通过它们对LLM的常识和推理能力的影响来衡量它们，并用经验证据说明了防御编辑攻击的难度。我们的发现表明，知识编辑技术正在出现的滥用风险危及低成本管理的安全性，以及以低成本管理作为新渠道传播错误信息或偏见的可行性。



## **46. Ask, Attend, Attack: A Effective Decision-Based Black-Box Targeted Attack for Image-to-Text Models**

询问、参与、攻击：针对图像到文本模型的有效基于决策的黑匣子定向攻击 cs.AI

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08989v1) [paper-pdf](http://arxiv.org/pdf/2408.08989v1)

**Authors**: Qingyuan Zeng, Zhenzhong Wang, Yiu-ming Cheung, Min Jiang

**Abstract**: While image-to-text models have demonstrated significant advancements in various vision-language tasks, they remain susceptible to adversarial attacks. Existing white-box attacks on image-to-text models require access to the architecture, gradients, and parameters of the target model, resulting in low practicality. Although the recently proposed gray-box attacks have improved practicality, they suffer from semantic loss during the training process, which limits their targeted attack performance. To advance adversarial attacks of image-to-text models, this paper focuses on a challenging scenario: decision-based black-box targeted attacks where the attackers only have access to the final output text and aim to perform targeted attacks. Specifically, we formulate the decision-based black-box targeted attack as a large-scale optimization problem. To efficiently solve the optimization problem, a three-stage process \textit{Ask, Attend, Attack}, called \textit{AAA}, is proposed to coordinate with the solver. \textit{Ask} guides attackers to create target texts that satisfy the specific semantics. \textit{Attend} identifies the crucial regions of the image for attacking, thus reducing the search space for the subsequent \textit{Attack}. \textit{Attack} uses an evolutionary algorithm to attack the crucial regions, where the attacks are semantically related to the target texts of \textit{Ask}, thus achieving targeted attacks without semantic loss. Experimental results on transformer-based and CNN+RNN-based image-to-text models confirmed the effectiveness of our proposed \textit{AAA}.

摘要: 虽然图像到文本模型在各种视觉语言任务中显示出了显著的进步，但它们仍然容易受到对手的攻击。现有的针对图像到文本模型的白盒攻击需要访问目标模型的体系结构、渐变和参数，导致实用性较低。最近提出的灰盒攻击虽然提高了实用性，但它们在训练过程中存在语义丢失问题，限制了它们的针对性攻击性能。为了推进图像到文本模型的对抗性攻击，本文重点研究了一个具有挑战性的场景：基于决策的黑箱定向攻击，攻击者只能访问最终的输出文本，并且目标是执行定向攻击。具体地说，我们将基于决策的黑盒定向攻击问题描述为一个大规模优化问题。为了有效地解决优化问题，提出了一个三阶段过程\textit{Ask}引导攻击者创建满足特定语义的目标文本。\textit{attend}识别图像中要攻击的关键区域，从而减少了后续\textit{攻击}的搜索空间。利用进化算法攻击与目标文本语义相关的关键区域，从而在不丢失语义的情况下实现目标攻击。在基于变压器和基于CNN+RNN的图文转换模型上的实验结果证实了该方法的有效性。



## **47. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08685v1) [paper-pdf](http://arxiv.org/pdf/2408.08685v1)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, especially for topology attacks, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attack. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph.

摘要: 图神经网络(GNN)很容易受到敌意干扰，尤其是对拓扑攻击，许多提高GNN稳健性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。



## **48. MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector**

MIA-Tuner：调整大型语言模型作为预训练文本检测器 cs.CL

code and dataset: https://github.com/wjfu99/MIA-Tuner

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2408.08661v1) [paper-pdf](http://arxiv.org/pdf/2408.08661v1)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: The increasing parameters and expansive dataset of large language models (LLMs) highlight the urgent demand for a technical solution to audit the underlying privacy risks and copyright issues associated with LLMs. Existing studies have partially addressed this need through an exploration of the pre-training data detection problem, which is an instance of a membership inference attack (MIA). This problem involves determining whether a given piece of text has been used during the pre-training phase of the target LLM. Although existing methods have designed various sophisticated MIA score functions to achieve considerable detection performance in pre-trained LLMs, how to achieve high-confidence detection and how to perform MIA on aligned LLMs remain challenging. In this paper, we propose MIA-Tuner, a novel instruction-based MIA method, which instructs LLMs themselves to serve as a more precise pre-training data detector internally, rather than design an external MIA score function. Furthermore, we design two instruction-based safeguards to respectively mitigate the privacy risks brought by the existing methods and MIA-Tuner. To comprehensively evaluate the most recent state-of-the-art LLMs, we collect a more up-to-date MIA benchmark dataset, named WIKIMIA-24, to replace the widely adopted benchmark WIKIMIA. We conduct extensive experiments across various aligned and unaligned LLMs over the two benchmark datasets. The results demonstrate that MIA-Tuner increases the AUC of MIAs from 0.7 to a significantly high level of 0.9.

摘要: 不断增加的参数和庞大的大型语言模型数据集突显了对技术解决方案的迫切需求，以审计与大型语言模型相关的潜在隐私风险和版权问题。现有的研究已经通过探索训练前数据检测问题部分地解决了这一需求，该问题是成员推理攻击(MIA)的一个实例。这个问题涉及确定在目标LLM的预训练阶段是否使用了给定的文本片段。虽然现有的方法已经设计了各种复杂的MIA评分函数来在预先训练的LLM中获得相当高的检测性能，但如何实现高置信度检测以及如何在对准的LLM上执行MIA仍然是具有挑战性的。在本文中，我们提出了一种新的基于指令的MIA方法MIA-Tuner，它指示LLMS本身在内部充当更精确的预训练数据检测器，而不是在外部设计MIA得分函数。此外，我们设计了两个基于指令的安全机制，分别缓解了现有方法和MIA-Tuner带来的隐私风险。为了全面评估最新的LLM，我们收集了一个更新的MIA基准数据集，名为WIKIMIA-24，以取代广泛采用的基准WIKIMIA。我们在两个基准数据集上对各种对齐和未对齐的LLM进行了广泛的实验。结果表明，MIA-Tuner将MIA的AUC从0.7提高到0.9的显著高水平。



## **49. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2407.06992v2) [paper-pdf](http://arxiv.org/pdf/2407.06992v2)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **50. ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages**

Tools Sword：跨越三个阶段揭示工具学习中大型语言模型的安全问题 cs.CL

Accepted by ACL 2024 Main Conference

**SubmitDate**: 2024-08-16    [abs](http://arxiv.org/abs/2402.10753v2) [paper-pdf](http://arxiv.org/pdf/2402.10753v2)

**Authors**: Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, Xuanjing Huang

**Abstract**: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present *ToolSword*, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing **malicious queries** and **jailbreak attacks** in the input stage, **noisy misdirection** and **risky cues** in the execution stage, and **harmful feedback** and **error conflicts** in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedback, which even GPT-4 is susceptible to. Moreover, we conduct further studies with the aim of fostering research on tool learning safety. The data is released in https://github.com/Junjie-Ye/ToolSword.

摘要: 工具学习被广泛认为是在现实世界场景中部署大型语言模型(LLM)的基本方法。虽然目前的研究主要强调利用工具来增强LLM，但它往往忽略了与其应用相关的新出现的安全考虑。为了填补这一空白，我们推出了*ToolSword*，这是一个全面的框架，致力于在工具学习中仔细调查与LLM相关的安全问题。具体地说，ToolSword为低层管理工具学习划分了六个安全场景，包括输入阶段的**恶意查询**和**越狱攻击**，执行阶段的**噪音误导**和**危险提示**，以及输出阶段的**有害反馈**和**错误冲突**。在11个开源和封闭源代码的LLM上进行的实验表明，工具学习中存在持久的安全挑战，例如处理有害的查询、使用危险的工具以及提供有害的反馈，这些都是GPT-4容易受到的。此外，我们还进行了进一步的研究，旨在促进对工具学习安全性的研究。数据以https://github.com/Junjie-Ye/ToolSword.格式发布



