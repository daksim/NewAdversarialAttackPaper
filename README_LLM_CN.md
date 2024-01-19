# Latest Large Language Model Attack Papers
**update at 2024-01-19 11:39:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**

万事俱备：越狱攻击的简单黑匣子方法 cs.CL

11 pages, 3 figures, 2 tables

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09798v1) [paper-pdf](http://arxiv.org/pdf/2401.09798v1)

**Authors**: Kazuhiro Takemoto

**Abstract**: Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study introduces a simple black-box method to effectively generate jailbreak prompts, overcoming the limitations of high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample safeguard-bypassing expressions. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The jailbreak prompts generated were naturally-worded and concise, suggesting they are less detectable. The results indicate that creating effective jailbreak prompts is simpler than previously considered, and black-box jailbreak attacks pose a more serious security threat.

摘要: 像ChatGPT这样的大型语言模型(LLM)面临着“越狱”的挑战，在这种情况下，安全措施会被绕过，产生道德上有害的提示。本研究介绍了一种简单的黑盒方法来有效地生成越狱提示，克服了现有方法存在的高复杂性和计算代价的局限性。该技术使用目标LLM本身迭代地将有害提示重写为无害的表达，该方法基于LLM可以直接采样绕过安全保护的表达的假设。通过对ChatGPT(GPT-3.5和GPT-4)和Gemini-Pro的实验证明，该方法在平均5次迭代内达到了80%以上的攻击成功率，并且在模型更新的情况下仍然有效。生成的越狱提示措辞自然，简明扼要，这表明它们不太容易被发现。结果表明，创建有效的越狱提示比之前认为的要简单，黑盒越狱攻击构成了更严重的安全威胁。



## **2. Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings**

大型语言模型横向鱼叉式网络钓鱼：大规模组织背景下的比较研究 cs.CR

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09727v1) [paper-pdf](http://arxiv.org/pdf/2401.09727v1)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted by LLMs. This study is a pioneering exploration into the use of Large Language Models (LLMs) for the creation of targeted lateral phishing emails, targeting a large tier 1 university's operation and workforce of approximately 9,000 individuals over an 11-month period. It also evaluates the capability of email filtering infrastructure to detect such LLM-generated phishing attempts, providing insights into their effectiveness and identifying potential areas for improvement. Based on our findings, we propose machine learning-based detection techniques for such emails to detect LLM-generated phishing emails that were missed by the existing infrastructure, with an F1-score of 98.96.

摘要: 钓鱼电子邮件的严重威胁由于LLMS可能产生高度针对性、个性化和自动化的鱼叉式钓鱼攻击而进一步加剧。与LLM协助的网络钓鱼有关的两个关键问题需要进一步调查：1)现有的横向网络钓鱼研究缺乏针对针对整个组织的大规模攻击的LLM集成的具体检查；2)当前的反网络钓鱼基础设施尽管得到了广泛的发展，但缺乏阻止LLM生成的攻击的能力，这可能会影响员工和IT安全事件管理。然而，进行这样的调查研究需要一个真实的环境，一个在常规业务运营期间运作的环境，一个反映大型组织基础设施复杂性的环境。这种设置还必须提供所需的灵活性，以促进不同的实验条件，特别是纳入由LLMS精心编制的钓鱼电子邮件。这项研究是对使用大型语言模型(LLM)创建有针对性的横向网络钓鱼电子邮件的开创性探索，目标是一所大型一线大学的运营和员工队伍，在11个月的时间里大约有9000人。它还评估了电子邮件过滤基础设施检测此类LLM生成的网络钓鱼尝试的能力，提供了对其有效性的见解，并确定了潜在的改进领域。基于我们的发现，我们提出了基于机器学习的此类电子邮件检测技术，以检测现有基础设施错过的LLM生成的钓鱼电子邮件，F1得分为98.96。



## **3. MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance**

MLLM-保护器：在不损害性能的情况下确保MLLM的安全 cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.02906v2) [paper-pdf](http://arxiv.org/pdf/2401.02906v2)

**Authors**: Renjie Pi, Tianyang Han, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, Tong Zhang

**Abstract**: The deployment of multimodal large language models (MLLMs) has brought forth a unique vulnerability: susceptibility to malicious attacks through visual inputs. We delve into the novel challenge of defending MLLMs against such attacks. We discovered that images act as a "foreign language" that is not considered during alignment, which can make MLLMs prone to producing harmful responses. Unfortunately, unlike the discrete tokens considered in text-based LLMs, the continuous nature of image signals presents significant alignment challenges, which poses difficulty to thoroughly cover the possible scenarios. This vulnerability is exacerbated by the fact that open-source MLLMs are predominantly fine-tuned on limited image-text pairs that is much less than the extensive text-based pretraining corpus, which makes the MLLMs more prone to catastrophic forgetting of their original abilities during explicit alignment tuning. To tackle these challenges, we introduce MLLM-Protector, a plug-and-play strategy combining a lightweight harm detector and a response detoxifier. The harm detector's role is to identify potentially harmful outputs from the MLLM, while the detoxifier corrects these outputs to ensure the response stipulates to the safety standards. This approach effectively mitigates the risks posed by malicious visual inputs without compromising the model's overall performance. Our results demonstrate that MLLM-Protector offers a robust solution to a previously unaddressed aspect of MLLM security.

摘要: 多模式大型语言模型(MLLMS)的部署带来了一个独特的漏洞：通过视觉输入易受恶意攻击。我们深入研究了保护MLLMS免受此类攻击的新挑战。我们发现，图像是一种“外语”，在对齐过程中没有考虑到这一点，这可能会使MLLM容易产生有害的反应。不幸的是，与基于文本的LLMS中考虑的离散令牌不同，图像信号的连续性质带来了巨大的对齐挑战，这使得很难完全覆盖可能的场景。开源MLLMS主要在有限的图文对上进行微调，这比基于大量文本的预训练语料库要少得多，这使得MLLMS在显式对齐调整期间更容易灾难性地忘记其原始能力，从而加剧了这一漏洞。为了应对这些挑战，我们引入了MLLM-Protector，这是一种结合了轻型伤害探测器和响应解毒器的即插即用策略。危害检测器的作用是识别MLLM的潜在有害输出，而解毒器纠正这些输出，以确保响应符合安全标准。这种方法有效地降低了恶意视觉输入带来的风险，而不会影响模型的整体性能。我们的结果表明，MLLM-Protector为MLLM安全的一个以前未解决的方面提供了一个健壮的解决方案。



## **4. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

攻坚战：如何评估越狱攻击在大型语言模型上的有效性 cs.CL

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09002v1) [paper-pdf](http://arxiv.org/pdf/2401.09002v1)

**Authors**: Dong shu, Mingyu Jin, Suiyuan Zhu, Beichen Wang, Zihao Zhou, Chong Zhang, Yongfeng Zhang

**Abstract**: In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation aligns with the baseline's trend while offering a more profound and detailed assessment. We believe that by accurately evaluating the effectiveness of attack prompts in the Jailbreak task, our work lays a solid foundation for assessing a wider array of similar or even more complex tasks in the realm of prompt injection, potentially revolutionizing this field.

摘要: 在我们的研究中，我们开创了一种新的方法来评估越狱攻击对大型语言模型(如GPT-4和LLaMa2)的有效性，不同于传统的专注于健壮性的二进制评估。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供了一个独特的视角，能够对攻击效果进行更全面和细微的评估，并使攻击者能够更好地了解他们的攻击提示。此外，我们还开发了专门为越狱任务量身定做的全面地面事实数据集。这一数据集不仅是我们当前研究的重要基准，而且还为未来的研究奠定了基础资源，使这一不断发展的领域能够进行一致和比较的分析。通过与传统评估方法的细致比较，我们发现我们的评估符合基线的趋势，同时提供了更深入和详细的评估。我们相信，通过准确评估越狱任务中攻击提示的有效性，我们的工作为评估快速注射领域中更广泛的类似甚至更复杂的任务奠定了坚实的基础，这可能会给这一领域带来革命性的变化。



## **5. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

低语像素：利用现代GPU中未初始化的寄存器访问 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).

摘要: 图形处理单元(GPU)已经超越了渲染图形的传统用例，如今也成为加速无处不在的非图形渲染任务的强大平台。一项突出的任务是神经网络的推理，它处理大量的个人数据，如音频、文本或图像。因此，GPU成为处理海量潜在机密数据的不可或缺的组件，这唤醒了安全研究人员的兴趣。这导致了近年来GPU中各种漏洞的发现。在本文中，我们发现了GPU中的另一个漏洞类别：我们发现一些GPU实现在着色器执行之前缺乏适当的寄存器初始化例程，导致先前执行的着色器内核的意外寄存器内容泄漏。我们展示了3家主要供应商的产品上存在上述漏洞-苹果、NVIDIA和高通。由于GPU固件中存在不透明的调度和寄存器重新映射算法，该漏洞对对手构成了独特的挑战，使泄漏数据的重建复杂化。为了说明该漏洞的实际影响，我们展示了如何解决这些挑战来攻击GPU上的各种工作负载。首先，我们展示了未初始化的寄存器如何泄漏由片段着色器处理的任意像素数据。在此基础上，对卷积神经网络(CNN)的中间数据进行了信息泄漏攻击，并给出了该攻击对大语言模型(LLM)输出的泄漏和重构能力。



## **6. IsamasRed: A Public Dataset Tracking Reddit Discussions on Israel-Hamas Conflict**

IsamasRed：一个追踪Reddit关于以色列和哈马斯冲突的公共数据集 cs.SI

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08202v1) [paper-pdf](http://arxiv.org/pdf/2401.08202v1)

**Authors**: Kai Chen, Zihao He, Keith Burghardt, Jingxin Zhang, Kristina Lerman

**Abstract**: The conflict between Israel and Palestinians significantly escalated after the October 7, 2023 Hamas attack, capturing global attention. To understand the public discourse on this conflict, we present a meticulously compiled dataset--IsamasRed--comprising nearly 400,000 conversations and over 8 million comments from Reddit, spanning from August 2023 to November 2023. We introduce an innovative keyword extraction framework leveraging a large language model to effectively identify pertinent keywords, ensuring a comprehensive data collection. Our initial analysis on the dataset, examining topics, controversy, emotional and moral language trends over time, highlights the emotionally charged and complex nature of the discourse. This dataset aims to enrich the understanding of online discussions, shedding light on the complex interplay between ideology, sentiment, and community engagement in digital spaces.

摘要: 2023年10月7日哈马斯袭击事件后，以色列和巴勒斯坦之间的冲突显著升级，引起了全球的关注。为了理解公众对这场冲突的讨论，我们提供了一个精心编制的数据集-IsamasRed--从2023年8月到2023年11月，包含来自Reddit的近40万次对话和800多万条评论。我们引入了一个创新的关键字提取框架，利用一个大型语言模型来有效地识别相关关键字，确保全面的数据收集。我们对数据集的初步分析，考察了话题、争议、情感和道德语言随着时间的推移的趋势，突显了话语的情绪化和复杂性。该数据集旨在丰富对在线讨论的理解，揭示数字空间中意识形态、情绪和社区参与之间的复杂相互作用。



## **7. MGTBench: Benchmarking Machine-Generated Text Detection**

MGTB：机器生成文本检测的基准测试 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2303.14822v3) [paper-pdf](http://arxiv.org/pdf/2303.14822v3)

**Authors**: Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, Yang Zhang

**Abstract**: Nowadays, powerful large language models (LLMs) such as ChatGPT have demonstrated revolutionary power in a variety of tasks. Consequently, the detection of machine-generated texts (MGTs) is becoming increasingly crucial as LLMs become more advanced and prevalent. These models have the ability to generate human-like language, making it challenging to discern whether a text is authored by a human or a machine. This raises concerns regarding authenticity, accountability, and potential bias. However, existing methods for detecting MGTs are evaluated using different model architectures, datasets, and experimental settings, resulting in a lack of a comprehensive evaluation framework that encompasses various methodologies. Furthermore, it remains unclear how existing detection methods would perform against powerful LLMs. In this paper, we fill this gap by proposing the first benchmark framework for MGT detection against powerful LLMs, named MGTBench. Extensive evaluations on public datasets with curated texts generated by various powerful LLMs such as ChatGPT-turbo and Claude demonstrate the effectiveness of different detection methods. Our ablation study shows that a larger number of words in general leads to better performance and most detection methods can achieve similar performance with much fewer training samples. Moreover, we delve into a more challenging task: text attribution. Our findings indicate that the model-based detection methods still perform well in the text attribution task. To investigate the robustness of different detection methods, we consider three adversarial attacks, namely paraphrasing, random spacing, and adversarial perturbations. We discover that these attacks can significantly diminish detection effectiveness, underscoring the critical need for the development of more robust detection methods.

摘要: 如今，强大的大型语言模型(LLM)，如ChatGPT，已经在各种任务中展示了革命性的力量。因此，随着LLMS变得更加先进和普遍，机器生成文本(MGTS)的检测变得越来越重要。这些模型具有生成类似人类的语言的能力，这使得辨别文本是由人还是由机器创作具有挑战性。这引发了人们对真实性、问责性和潜在偏见的担忧。然而，现有的检测MGTS的方法是使用不同的模型体系结构、数据集和实验设置来评估的，导致缺乏包含各种方法的全面评估框架。此外，目前尚不清楚现有的检测方法将如何对抗强大的LLMS。在本文中，我们通过提出第一个针对强大的LLMS的MGT检测的基准框架来填补这一空白，称为MGTB。在公共数据集上的广泛评估与各种强大的LLMS生成的精选文本，如ChatGPT-Turbo和Claude证明了不同检测方法的有效性。我们的烧蚀研究表明，通常情况下，单词数量越多，性能越好，大多数检测方法都可以在更少的训练样本下获得类似的性能。此外，我们还深入研究了一项更具挑战性的任务：文本归因。我们的研究结果表明，基于模型的检测方法在文本归因任务中仍然表现良好。为了考察不同检测方法的稳健性，我们考虑了三种对抗性攻击，即释义攻击、随机间隔攻击和对抗性扰动攻击。我们发现，这些攻击会显著降低检测效率，这突显了开发更健壮的检测方法的迫切需要。



## **8. Traces of Memorisation in Large Language Models for Code**

代码的大型语言模型中的记忆痕迹 cs.CR

ICSE 2024 Research Track

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2312.11658v2) [paper-pdf](http://arxiv.org/pdf/2312.11658v2)

**Authors**: Ali Al-Kaswan, Maliheh Izadi, Arie van Deursen

**Abstract**: Large language models have gained significant popularity because of their ability to generate human-like text and potential applications in various fields, such as Software Engineering. Large language models for code are commonly trained on large unsanitised corpora of source code scraped from the internet. The content of these datasets is memorised and can be extracted by attackers with data extraction attacks. In this work, we explore memorisation in large language models for code and compare the rate of memorisation with large language models trained on natural language. We adopt an existing benchmark for natural language and construct a benchmark for code by identifying samples that are vulnerable to attack. We run both benchmarks against a variety of models, and perform a data extraction attack. We find that large language models for code are vulnerable to data extraction attacks, like their natural language counterparts. From the training data that was identified to be potentially extractable we were able to extract 47% from a CodeGen-Mono-16B code completion model. We also observe that models memorise more, as their parameter count grows, and that their pre-training data are also vulnerable to attack. We also find that data carriers are memorised at a higher rate than regular code or documentation and that different model architectures memorise different samples. Data leakage has severe outcomes, so we urge the research community to further investigate the extent of this phenomenon using a wider range of models and extraction techniques in order to build safeguards to mitigate this issue.

摘要: 大型语言模型因其生成类似人类的文本的能力以及在软件工程等各个领域的潜在应用而广受欢迎。代码的大型语言模型通常是在从互联网上刮来的大量未经清理的源代码语料库上进行训练的。这些数据集的内容是被记忆的，可以被具有数据提取攻击的攻击者提取出来。在这项工作中，我们探索了代码在大型语言模型中的记忆，并将记忆速度与自然语言训练的大型语言模型进行了比较。我们采用现有的自然语言基准，并通过识别易受攻击的样本来构建代码基准。我们针对各种模型运行这两个基准测试，并执行数据提取攻击。我们发现代码的大型语言模型很容易受到数据提取攻击，就像它们的自然语言模型一样。从被确定为潜在可提取的训练数据中，我们能够从CodeGen-Mono-16B代码完成模型中提取47%。我们还观察到，随着参数数量的增加，模型记住的更多，而且它们的预训练数据也很容易受到攻击。我们还发现，数据载体的记忆速度高于常规代码或文档，并且不同的模型体系结构记忆的样本不同。数据泄露具有严重的后果，因此我们敦促研究界使用更广泛的模型和提取技术进一步调查这一现象的程度，以便建立保障措施来缓解这一问题。



## **9. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语种机器文本检测中的作者身份混淆 cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07867v1) [paper-pdf](http://arxiv.org/pdf/2401.07867v1)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.

摘要: 最新的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，所有测试的声学方法都能在所有测试的语言中造成检测逃避，其中同形文字攻击尤其成功。



## **10. Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications**

Signed-Prompt：一种防止LLM集成应用的即时注入攻击的新方法 cs.CR

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07612v1) [paper-pdf](http://arxiv.org/pdf/2401.07612v1)

**Authors**: Xuchen Suo

**Abstract**: The critical challenge of prompt injection attacks in Large Language Models (LLMs) integrated applications, a growing concern in the Artificial Intelligence (AI) field. Such attacks, which manipulate LLMs through natural language inputs, pose a significant threat to the security of these applications. Traditional defense strategies, including output and input filtering, as well as delimiter use, have proven inadequate. This paper introduces the 'Signed-Prompt' method as a novel solution. The study involves signing sensitive instructions within command segments by authorized users, enabling the LLM to discern trusted instruction sources. The paper presents a comprehensive analysis of prompt injection attack patterns, followed by a detailed explanation of the Signed-Prompt concept, including its basic architecture and implementation through both prompt engineering and fine-tuning of LLMs. Experiments demonstrate the effectiveness of the Signed-Prompt method, showing substantial resistance to various types of prompt injection attacks, thus validating its potential as a robust defense strategy in AI security.

摘要: 大型语言模型（LLM）集成应用程序中的即时注入攻击是人工智能（AI）领域日益关注的关键挑战。这种通过自然语言输入操纵LLM的攻击对这些应用程序的安全性构成了重大威胁。传统的防御策略，包括输出和输入过滤，以及过滤，已被证明是不够的。本文介绍了一种新的解决方案，“签名提示”的方法。该研究涉及由授权用户在命令段内签署敏感指令，使LLM能够识别可信的指令源。本文提出了一个全面的分析提示注入攻击模式，然后详细解释了签名提示的概念，包括其基本架构和实现，通过提示工程和微调的LLM。实验证明了Signed-Prompt方法的有效性，对各种类型的提示注入攻击表现出很大的抵抗力，从而验证了其作为AI安全中强大防御策略的潜力。



## **11. Stability Analysis of ChatGPT-based Sentiment Analysis in AI Quality Assurance**

人工智能质量保证中基于ChatGPT情感分析的稳定性分析 cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07441v1) [paper-pdf](http://arxiv.org/pdf/2401.07441v1)

**Authors**: Tinghui Ouyang, AprilPyone MaungMaung, Koichi Konishi, Yoshiki Seo, Isao Echizen

**Abstract**: In the era of large AI models, the complex architecture and vast parameters present substantial challenges for effective AI quality management (AIQM), e.g. large language model (LLM). This paper focuses on investigating the quality assurance of a specific LLM-based AI product--a ChatGPT-based sentiment analysis system. The study delves into stability issues related to both the operation and robustness of the expansive AI model on which ChatGPT is based. Experimental analysis is conducted using benchmark datasets for sentiment analysis. The results reveal that the constructed ChatGPT-based sentiment analysis system exhibits uncertainty, which is attributed to various operational factors. It demonstrated that the system also exhibits stability issues in handling conventional small text attacks involving robustness.

摘要: 在大型人工智能模型的时代，复杂的体系结构和庞大的参数对有效的人工智能质量管理(AIQM)提出了巨大的挑战，例如大型语言模型(LLM)。本文重点研究了一个具体的基于LLM的人工智能产品--基于ChatGPT的情感分析系统的质量保证。这项研究深入探讨了与ChatGPT所基于的扩展人工智能模型的操作和健壮性相关的稳定性问题。使用用于情感分析的基准数据集进行了实验分析。结果表明，构建的基于ChatGPT的情感分析系统存在不确定性，这归因于各种操作因素。结果表明，该系统在处理涉及健壮性的常规小文本攻击时也存在稳定性问题。



## **12. Advancing TTP Analysis: Harnessing the Power of Encoder-Only and Decoder-Only Language Models with Retrieval Augmented Generation**

高级TTP分析：利用仅编码者和仅解码者的语言模型和检索增强生成的能力 cs.CR

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.00280v2) [paper-pdf](http://arxiv.org/pdf/2401.00280v2)

**Authors**: Reza Fayyazi, Rozhina Taghdimi, Shanchieh Jay Yang

**Abstract**: Tactics, Techniques, and Procedures (TTPs) outline the methods attackers use to exploit vulnerabilities. The interpretation of TTPs in the MITRE ATT&CK framework can be challenging for cybersecurity practitioners due to presumed expertise, complex dependencies, and inherent ambiguity. Meanwhile, advancements with Large Language Models (LLMs) have led to recent surge in studies exploring its uses in cybersecurity operations. This leads us to question how well encoder-only (e.g., RoBERTa) and decoder-only (e.g., GPT-3.5) LLMs can comprehend and summarize TTPs to inform analysts of the intended purposes (i.e., tactics) of a cyberattack procedure. The state-of-the-art LLMs have shown to be prone to hallucination by providing inaccurate information, which is problematic in critical domains like cybersecurity. Therefore, we propose the use of Retrieval Augmented Generation (RAG) techniques to extract relevant contexts for each cyberattack procedure for decoder-only LLMs (without fine-tuning). We further contrast such approach against supervised fine-tuning (SFT) of encoder-only LLMs. Our results reveal that both the direct-use of decoder-only LLMs (i.e., its pre-trained knowledge) and the SFT of encoder-only LLMs offer inaccurate interpretation of cyberattack procedures. Significant improvements are shown when RAG is used for decoder-only LLMs, particularly when directly relevant context is found. This study further sheds insights on the limitations and capabilities of using RAG for LLMs in interpreting TTPs.

摘要: 战术、技术和过程(TTP)概述了攻击者用来利用漏洞的方法。由于假定的专业知识、复杂的依赖关系和固有的模糊性，MITRE ATT&CK框架中对TTP的解释可能会对网络安全从业者构成挑战。与此同时，大型语言模型(LLM)的进步导致了最近探索其在网络安全行动中的应用的研究激增。这导致我们质疑仅编码器(例如Roberta)和仅解码器(例如GPT-3.5)的LLM能够在多大程度上理解和总结TTP以告知分析师网络攻击过程的预期目的(即战术)。最先进的LLM通过提供不准确的信息而容易产生幻觉，这在网络安全等关键领域是有问题的。因此，我们提出使用检索增强生成(RAG)技术来提取仅针对解码器的LLMS的每个网络攻击过程的相关上下文(无需微调)。我们进一步将这种方法与仅编码器的LLM的有监督微调(SFT)进行了对比。我们的结果表明，直接使用仅解码的LLM(即其预先训练的知识)和仅编码的LLM的SFT都提供了对网络攻击过程的不准确解释。当RAG被用于仅解码器的LLM时，尤其是当找到直接相关的上下文时，显示出显著的改进。这项研究进一步揭示了使用RAG对LLMS进行TTP解释的局限性和能力。



## **13. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

约翰尼如何说服低层管理人员越狱：通过将低层管理人员人性化来挑战人工智能安全的再思考 cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06373v1) [paper-pdf](http://arxiv.org/pdf/2401.06373v1)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs

摘要: 大多数传统的人工智能安全研究都将人工智能模型视为机器，并集中在安全专家开发的以算法为重点的攻击上。随着大型语言模型(LLM)变得越来越普遍和有能力，非专家用户也可能在日常交互中带来风险。本文介绍了一种新的视角，将越狱LLMS作为类人类的沟通者，来探索日常语言交互和人工智能安全之间被忽视的交集。具体地说，我们研究如何说服LLMS越狱。首先，我们提出了一种源于数十年社会科学研究的说服分类法。然后，我们应用分类法自动生成可解释的说服性对抗性提示(PAP)来越狱LLM。结果表明，说服显著提高了所有风险类别的越狱性能：PAP在Llama 2-7b Chat、GPT-3.5和GPT-4上的攻击成功率在10美元的试验中始终保持在92美元以上，超过了最近针对算法的攻击。在防御方面，我们探索了各种对抗PAP和的机制，发现了现有防御措施中的显著差距，并倡导从更根本上缓解高度互动的LLM



## **14. Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender**

意图分析提示使大型语言模型成为优秀的越狱捍卫者 cs.CL

9 pages, 5 figures

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06561v1) [paper-pdf](http://arxiv.org/pdf/2401.06561v1)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreaks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis Prompting (IAPrompt). The principle behind is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that IAPrompt could consistently and significantly reduce the harmfulness in response (averagely -46.5% attack success rate) and maintain the general helpfulness. Further analyses present some insights into how our method works. To facilitate reproducibility, We release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis

摘要: 使大型语言模型(LLM)与人类价值观保持一致，尤其是在面对秘密和复杂的越狱时，这是一个艰巨的挑战。在本研究中，我们提出了一种简单而高效的防御策略，即意图分析提示(IAPrompt)。其背后的原理是通过一个两个阶段的过程触发LLMS内在的自我纠正和提高能力：1)基本意图分析，2)政策一致的反应。值得注意的是，IAPrompt是一种仅限推理的方法，因此可以在不影响其有用性的情况下增强LLMS的安全性。在Vicuna、ChatGLM、MPT、DeepSeek和GPT-3.5上对SAP200和DAN基准测试的广泛实验表明，IAPrompt可以持续且显著地降低响应的危害性(平均-46.5%的攻击成功率)，并保持总体帮助。进一步的分析为我们的方法是如何工作的提供了一些见解。为了便于重现，我们在https://github.com/alphadl/SafeLLM_with_IntentionAnalysis上发布了我们的代码和脚本



## **15. Universal Vulnerabilities in Large Language Models: In-context Learning Backdoor Attacks**

大型语言模型中的通用漏洞：情景学习后门攻击 cs.CL

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.05949v2) [paper-pdf](http://arxiv.org/pdf/2401.05949v2)

**Authors**: Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, Jinming Wen

**Abstract**: In-context learning, a paradigm bridging the gap between pre-training and fine-tuning, has demonstrated high efficacy in several NLP tasks, especially in few-shot settings. Unlike traditional fine-tuning methods, in-context learning adapts pre-trained models to unseen tasks without updating any parameters. Despite being widely applied, in-context learning is vulnerable to malicious attacks. In this work, we raise security concerns regarding this paradigm. Our studies demonstrate that an attacker can manipulate the behavior of large language models by poisoning the demonstration context, without the need for fine-tuning the model. Specifically, we have designed a new backdoor attack method, named ICLAttack, to target large language models based on in-context learning. Our method encompasses two types of attacks: poisoning demonstration examples and poisoning prompts, which can make models behave in accordance with predefined intentions. ICLAttack does not require additional fine-tuning to implant a backdoor, thus preserving the model's generality. Furthermore, the poisoned examples are correctly labeled, enhancing the natural stealth of our attack method. Extensive experimental results across several language models, ranging in size from 1.3B to 40B parameters, demonstrate the effectiveness of our attack method, exemplified by a high average attack success rate of 95.0% across the three datasets on OPT models. Our findings highlight the vulnerabilities of language models, and we hope this work will raise awareness of the possible security threats associated with in-context learning.

摘要: 情境学习是一种弥合预训练和微调之间差距的范式，在几个NLP任务中表现出了很高的效率，特别是在少数情况下。与传统的微调方法不同，情景学习使预先训练的模型适应未知的任务，而不需要更新任何参数。尽管情景学习被广泛应用，但它很容易受到恶意攻击。在这项工作中，我们提出了对此范式的安全担忧。我们的研究表明，攻击者可以通过毒化演示上下文来操纵大型语言模型的行为，而不需要对模型进行微调。具体地说，我们设计了一种新的后门攻击方法ICLAttack，以基于上下文学习的大型语言模型为目标。我们的方法包括两种类型的攻击：中毒演示示例和中毒提示，这可以使模型的行为符合预定义的意图。ICLAttack不需要额外的微调来植入后门，从而保持了模型的通用性。此外，有毒的例子被正确地标记，增强了我们攻击方法的自然隐蔽性。在几个语言模型上的广泛实验结果，从1.3B到40B参数的大小，证明了我们的攻击方法的有效性，例如在OPT模型上的三个数据集上的高平均攻击成功率为95.0%。我们的发现突显了语言模型的脆弱性，我们希望这项工作将提高人们对与情景学习相关的潜在安全威胁的认识。



## **16. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

朝向稳健剪枝：一种自适应的语言模型知识保留剪枝策略 cs.CL

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2310.13191v3) [paper-pdf](http://arxiv.org/pdf/2310.13191v3)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

摘要: 修剪目标最近已经超越了语言模型中的精确度和稀疏性，扩展到了健壮性。尽管如此，现有的方法在不断增加模型稀疏性的同时努力增强对敌对攻击的鲁棒性，并且需要重新训练过程。随着人类步入大型语言模型时代，这些问题变得日益突出。本文提出语言模型的稳健性与它们所包含的预训练知识的程度成正比。因此，我们提出了一种训练后剪枝策略，旨在忠实地复制密集语言模型的嵌入空间和特征空间，目的是在剪枝过程中保存更多的预先训练的知识。在这种设置中，每一层的重建误差不仅源于自身，还包括来自前几层的累积误差，然后进行自适应校正。与其他最先进的基线相比，我们的方法在精确度、稀疏性、健壮性和剪枝成本之间表现出了更好的平衡，在数据集Sst2、IMDB和AgNews上使用ERT，标志着在语言模型中朝着健壮剪枝迈出了重要的一步。



## **17. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

LimeAttack：文本硬标签对抗性攻击的局部可解释方法 cs.CL

18 pages, 38th AAAI Main Track

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2308.00319v2) [paper-pdf](http://arxiv.org/pdf/2308.00319v2)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.

摘要: 自然语言处理模型很容易受到敌意例子的影响。以前的文本对抗性攻击采用梯度或置信度分数来计算单词重要性排名并生成对抗性实例。然而，这些信息在现实世界中是不可用的。因此，我们将重点放在一种更现实和更具挑战性的环境中，即硬标签攻击，在这种情况下，攻击者只能查询模型并获得离散的预测标签。现有的硬标签攻击算法倾向于通过随机替换来初始化对抗性样本，然后利用复杂的启发式算法来优化对抗性扰动。这些方法需要大量的模型查询，攻击成功率受对手初始化的制约。本文提出了一种新的硬标签攻击算法LimeAttack，该算法利用一种局部可解释的方法来近似单词重要性排序，然后采用波束搜索来寻找最优解。大量实验表明，在相同的查询开销下，LimeAttack的攻击性能优于现有的硬标签攻击。此外，我们评估了LimeAttack在大型语言模型上的有效性，结果表明，对抗性例子仍然是对大型语言模型的重大威胁。LimeAttack生成的对抗性实例具有很强的可移植性，有效地提高了对抗性训练中模型的稳健性。



## **18. Jatmo: Prompt Injection Defense by Task-Specific Finetuning**

Jatmo：通过特定于任务的微调实现快速注入防御 cs.CR

24 pages, 6 figures

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2312.17673v2) [paper-pdf](http://arxiv.org/pdf/2312.17673v2)

**Authors**: Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, David Wagner

**Abstract**: Large Language Models (LLMs) are attracting significant research attention due to their instruction-following abilities, allowing users and developers to leverage LLMs for a variety of tasks. However, LLMs are vulnerable to prompt-injection attacks: a class of attacks that hijack the model's instruction-following abilities, changing responses to prompts to undesired, possibly malicious ones. In this work, we introduce Jatmo, a method for generating task-specific models resilient to prompt-injection attacks. Jatmo leverages the fact that LLMs can only follow instructions once they have undergone instruction tuning. It harnesses a teacher instruction-tuned model to generate a task-specific dataset, which is then used to fine-tune a base model (i.e., a non-instruction-tuned model). Jatmo only needs a task prompt and a dataset of inputs for the task: it uses the teacher model to generate outputs. For situations with no pre-existing datasets, Jatmo can use a single example, or in some cases none at all, to produce a fully synthetic dataset. Our experiments on seven tasks show that Jatmo models provide similar quality of outputs on their specific task as standard LLMs, while being resilient to prompt injections. The best attacks succeeded in less than 0.5% of cases against our models, versus 87% success rate against GPT-3.5-Turbo. We release Jatmo at https://github.com/wagner-group/prompt-injection-defense.

摘要: 大型语言模型(LLM)由于其遵循指令的能力而吸引了大量的研究关注，使用户和开发人员能够利用LLM来执行各种任务。然而，LLM很容易受到即时注入攻击：这是一类劫持模型的指令遵循能力的攻击，将对提示的响应更改为不受欢迎的、可能是恶意的提示。在这项工作中，我们介绍了Jatmo，一种生成对快速注入攻击具有弹性的特定任务模型的方法。Jatmo利用了这样一个事实，即LLM只有在经过指令调优后才能遵循指令。它利用教师指令调整的模型来生成特定于任务的数据集，然后使用该数据集来微调基本模型(即，非指令调整的模型)。Jatmo只需要任务提示符和任务输入的数据集：它使用教师模型来生成输出。对于没有预先存在的数据集的情况，Jatmo可以使用单个示例，或者在某些情况下根本不使用任何示例来生成完全合成的数据集。我们在七个任务上的实验表明，Jatmo模型在其特定任务中提供的输出质量与标准LLM相似，同时对快速注入具有弹性。在针对我们的模型的情况下，最好的攻击成功率不到0.5%，而针对GPT-3.5-Turbo的成功率为87%。我们在https://github.com/wagner-group/prompt-injection-defense.发布了贾特莫



## **19. The Stronger the Diffusion Model, the Easier the Backdoor: Data Poisoning to Induce Copyright Breaches Without Adjusting Finetuning Pipeline**

扩散模型越强，后门越多：不调整微调管道的数据中毒引发版权泄露 cs.CR

This study reveals that by subtly inserting non-copyright-infringing  poisoning data into a diffusion model's training dataset, it's possible to  trigger the model to generate copyrighted content, highlighting  vulnerabilities in current copyright protection strategies

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.04136v1) [paper-pdf](http://arxiv.org/pdf/2401.04136v1)

**Authors**: Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, Kenji Kawaguchi

**Abstract**: The commercialization of diffusion models, renowned for their ability to generate high-quality images that are often indistinguishable from real ones, brings forth potential copyright concerns. Although attempts have been made to impede unauthorized access to copyrighted material during training and to subsequently prevent DMs from generating copyrighted images, the effectiveness of these solutions remains unverified. This study explores the vulnerabilities associated with copyright protection in DMs by introducing a backdoor data poisoning attack (SilentBadDiffusion) against text-to-image diffusion models. Our attack method operates without requiring access to or control over the diffusion model's training or fine-tuning processes; it merely involves the insertion of poisoning data into the clean training dataset. This data, comprising poisoning images equipped with prompts, is generated by leveraging the powerful capabilities of multimodal large language models and text-guided image inpainting techniques. Our experimental results and analysis confirm the method's effectiveness. By integrating a minor portion of non-copyright-infringing stealthy poisoning data into the clean dataset-rendering it free from suspicion-we can prompt the finetuned diffusion models to produce copyrighted content when activated by specific trigger prompts. These findings underline potential pitfalls in the prevailing copyright protection strategies and underscore the necessity for increased scrutiny and preventative measures against the misuse of DMs.

摘要: 扩散模型的商业化以其生成高质量图像的能力而闻名，这些图像往往与真实图像难以区分，这带来了潜在的版权问题。虽然已经作出努力，以阻止未经授权的访问受版权保护的材料在培训期间，并随后防止DM生成受版权保护的图像，这些解决方案的有效性仍然没有得到验证。本研究探讨与版权保护DM的漏洞，通过引入后门数据中毒攻击（SilentBadDiffusion）对文本到图像的扩散模型。我们的攻击方法不需要访问或控制扩散模型的训练或微调过程;它只涉及将中毒数据插入干净的训练数据集。这些数据，包括配备了提示的中毒图像，是通过利用多模态大型语言模型和文本引导图像修复技术的强大功能生成的。我们的实验结果和分析证实了该方法的有效性。通过将一小部分非侵犯版权的隐形中毒数据集成到干净的网络中，使其免受怀疑，我们可以在特定的触发提示激活时，提示微调的扩散模型产生受版权保护的内容。这些发现突显了现行版权保护战略中的潜在陷阱，并强调有必要加强审查和预防措施，防止滥用DM。



## **20. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型（LLM）的评估对于评估其性能和减轻潜在的安全风险至关重要。在本文中，我们介绍了一个统一的库来评估LLM。它由几个易于研究人员使用和扩展的关键组件组成：提示构建，提示工程，数据集和模型加载，对抗性提示攻击，动态评估协议和分析工具。ObjectiveBench被设计成一个开放、通用和灵活的代码库，用于研究目的，可以促进创建新基准、部署下游应用程序和设计新评估协议的原始研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续得到支持。



## **21. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整的定向攻击 cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。本文提出了一种新颖实用的灰盒攻击方案，即攻击者只能访问受害者LVLM的可视编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大型语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(InstructTA)，对具有高可转移性的LVLMS进行定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高可转移性，我们用转译自LLM的指令扩充了指令$\boldSymbol{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。



## **22. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

从网络威胁情报报告中挖掘时态攻击模式 cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.

摘要: 防御网络攻击需要从业者对高级别的对手行为进行操作。网络威胁情报（CTI）报告对过去的网络攻击事件描述了恶意行为的时间链。为了避免重复的网络攻击事件，从业人员必须主动识别和防御重复的行动链-我们称之为时间攻击模式。自动挖掘行为之间的模式提供了关于过去网络攻击的对手行为的结构化和可操作的信息。本文的目标是通过从网络威胁情报报告中挖掘时间攻击模式，帮助安全从业人员优先考虑和主动防御网络攻击。为此，我们提出了ChronoCTI，这是一个自动化的管道，用于从过去网络攻击的网络威胁情报（CTI）报告中挖掘时间攻击模式。为了构建ChronoCTI，我们构建了时间攻击模式的真实数据集，并应用了最先进的大型语言模型、自然语言处理和机器学习技术。我们将ChronoCTI应用于一组713份CTI报告，其中我们确定了124种时间攻击模式-我们将其分为9种模式类别。我们发现，最普遍的模式类别是欺骗受害者用户执行恶意代码发起攻击，然后绕过受害者网络中的反恶意软件系统。根据观察到的模式，我们建议组织对用户进行网络安全最佳实践培训，引入功能有限的不可变操作系统，并实施多用户身份验证。此外，我们提倡从业人员利用ChronoCTI的自动挖掘功能，并针对反复出现的攻击模式设计对策。



## **23. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

安全和性能，为什么不能两者兼而有之呢？针对AI软件部署异构性攻击的双目标优化模型压缩 cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.

摘要: 人工智能(AI)软件中的深度学习模型的规模正在迅速增长，阻碍了在资源受限的设备(如智能手机)上的大规模部署。为了缓解这个问题，人工智能软件压缩起到了至关重要的作用，其目标是在保持高性能的同时压缩模型大小。然而，大模型中的固有缺陷可能会被压缩的模型继承。这样的缺陷很容易被攻击者利用，因为压缩模型通常部署在大量设备中，而没有足够的保护。在本文中，我们旨在从安全-性能联合优化的角度解决安全模型压缩问题。具体地说，受软件工程中测试驱动开发(TDD)范式的启发，我们提出了一个称为SafeCompress的测试驱动稀疏训练框架。通过将攻击机制模拟为安全测试，SafeCompress可以按照动态稀疏训练范式自动将大模型压缩为小模型。然后，考虑到两种典型的异构性攻击机制，即黑盒成员关系推理攻击和白盒成员关系推理攻击，我们开发了两个具体的实例：BMIA-SafeCompress和WMIA-SafeCompress。此外，我们实现了另一个实例MMIA-SafeCompress，通过扩展SafeCompress来防御对手同时进行黑盒和白盒成员推理攻击的情况。我们在计算机视觉和自然语言处理任务的五个数据集上进行了广泛的实验。结果表明，该框架具有较好的通用性和有效性。我们还讨论了如何使SafeCompress适应除成员推理攻击之外的其他攻击，展示了SafeCompress的灵活性。



## **24. Detection and Defense Against Prominent Attacks on Preconditioned LLM-Integrated Virtual Assistants**

对预置LLM集成虚拟助理的显著攻击的检测和防御 cs.CR

Accepted to be published in the Proceedings of the 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00994v1) [paper-pdf](http://arxiv.org/pdf/2401.00994v1)

**Authors**: Chun Fai Chan, Daniel Wankit Yip, Aysan Esmradi

**Abstract**: The emergence of LLM (Large Language Model) integrated virtual assistants has brought about a rapid transformation in communication dynamics. During virtual assistant development, some developers prefer to leverage the system message, also known as an initial prompt or custom prompt, for preconditioning purposes. However, it is important to recognize that an excessive reliance on this functionality raises the risk of manipulation by malicious actors who can exploit it with carefully crafted prompts. Such malicious manipulation poses a significant threat, potentially compromising the accuracy and reliability of the virtual assistant's responses. Consequently, safeguarding the virtual assistants with detection and defense mechanisms becomes of paramount importance to ensure their safety and integrity. In this study, we explored three detection and defense mechanisms aimed at countering attacks that target the system message. These mechanisms include inserting a reference key, utilizing an LLM evaluator, and implementing a Self-Reminder. To showcase the efficacy of these mechanisms, they were tested against prominent attack techniques. Our findings demonstrate that the investigated mechanisms are capable of accurately identifying and counteracting the attacks. The effectiveness of these mechanisms underscores their potential in safeguarding the integrity and reliability of virtual assistants, reinforcing the importance of their implementation in real-world scenarios. By prioritizing the security of virtual assistants, organizations can maintain user trust, preserve the integrity of the application, and uphold the high standards expected in this era of transformative technologies.

摘要: LLM(Large Language Model，大型语言模型)集成虚拟助手的出现，带来了交流动力学的快速变革。在虚拟助手开发期间，一些开发人员更喜欢利用系统消息(也称为初始提示或自定义提示)进行预条件处理。但是，重要的是要认识到，过度依赖此功能会增加恶意攻击者操纵该功能的风险，恶意攻击者可以通过精心设计的提示来利用该功能。这种恶意操作构成了重大威胁，可能会损害虚拟助理响应的准确性和可靠性。因此，使用检测和防御机制来保护虚拟助理，对于确保其安全性和完整性至关重要。在本研究中，我们探索了三种检测和防御机制，旨在对抗以系统消息为目标的攻击。这些机制包括插入引用关键字、使用LLM求值器以及实现自我提醒。为了展示这些机制的有效性，他们针对突出的攻击技术进行了测试。我们的研究结果表明，所研究的机制能够准确地识别和对抗攻击。这些机制的有效性突出了它们在保障虚拟助理的完整性和可靠性方面的潜力，加强了在现实世界情景中执行这些机制的重要性。通过优先考虑虚拟助理的安全性，组织可以维护用户信任、维护应用程序的完整性，并保持在这个变革性技术时代所期望的高标准。



## **25. A Novel Evaluation Framework for Assessing Resilience Against Prompt Injection Attacks in Large Language Models**

一种新的评估大型语言模型抗即时注入攻击能力的评估框架 cs.CR

Accepted to be published in the Proceedings of The 10th IEEE CSDE  2023, the Asia-Pacific Conference on Computer Science and Data Engineering  2023

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00991v1) [paper-pdf](http://arxiv.org/pdf/2401.00991v1)

**Authors**: Daniel Wankit Yip, Aysan Esmradi, Chun Fai Chan

**Abstract**: Prompt injection attacks exploit vulnerabilities in large language models (LLMs) to manipulate the model into unintended actions or generate malicious content. As LLM integrated applications gain wider adoption, they face growing susceptibility to such attacks. This study introduces a novel evaluation framework for quantifying the resilience of applications. The framework incorporates innovative techniques designed to ensure representativeness, interpretability, and robustness. To ensure the representativeness of simulated attacks on the application, a meticulous selection process was employed, resulting in 115 carefully chosen attacks based on coverage and relevance. For enhanced interpretability, a second LLM was utilized to evaluate the responses generated from these simulated attacks. Unlike conventional malicious content classifiers that provide only a confidence score, the LLM-based evaluation produces a score accompanied by an explanation, thereby enhancing interpretability. Subsequently, a resilience score is computed by assigning higher weights to attacks with greater impact, thus providing a robust measurement of the application resilience. To assess the framework's efficacy, it was applied on two LLMs, namely Llama2 and ChatGLM. Results revealed that Llama2, the newer model exhibited higher resilience compared to ChatGLM. This finding substantiates the effectiveness of the framework, aligning with the prevailing notion that newer models tend to possess greater resilience. Moreover, the framework exhibited exceptional versatility, requiring only minimal adjustments to accommodate emerging attack techniques and classifications, thereby establishing itself as an effective and practical solution. Overall, the framework offers valuable insights that empower organizations to make well-informed decisions to fortify their applications against potential threats from prompt injection.

摘要: 提示注入攻击利用大型语言模型(LLM)中的漏洞将模型操纵为意外操作或生成恶意内容。随着LLM集成应用程序得到更广泛的采用，它们面临着越来越容易受到此类攻击的风险。这项研究介绍了一种新的评估框架，用于量化应用程序的弹性。该框架结合了旨在确保代表性、可解释性和健壮性的创新技术。为了确保对应用程序的模拟攻击的代表性，采用了精心选择的过程，根据覆盖范围和相关性精心选择了115个攻击。为了增强可解释性，使用了第二个LLM来评估这些模拟攻击生成的响应。与仅提供置信度分数的传统恶意内容分类器不同，基于LLM的评估会生成伴随解释的分数，从而增强可解释性。随后，通过将更高的权重分配给影响更大的攻击来计算弹性分数，从而提供对应用程序弹性的稳健测量。为了评估该框架的有效性，将其应用于两个LLM上，即Llama2和ChatGLM。结果表明，与ChatGLM相比，较新的模型Llama2表现出更高的弹性。这一发现证实了该框架的有效性，与流行的观点一致，即较新的模型往往具有更强的弹性。此外，该框架显示出非凡的多功能性，只需进行最小的调整即可适应新出现的攻击技术和分类，从而确立其本身是一种有效和实用的解决方案。总体而言，该框架提供了有价值的见解，使组织能够做出明智的决策，以加强其应用程序免受即时注入的潜在威胁。



## **26. Opening A Pandora's Box: Things You Should Know in the Era of Custom GPTs**

打开潘多拉的盒子：定制GPT时代你应该知道的事情 cs.CR

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00905v1) [paper-pdf](http://arxiv.org/pdf/2401.00905v1)

**Authors**: Guanhong Tao, Siyuan Cheng, Zhuo Zhang, Junmin Zhu, Guangyu Shen, Xiangyu Zhang

**Abstract**: The emergence of large language models (LLMs) has significantly accelerated the development of a wide range of applications across various fields. There is a growing trend in the construction of specialized platforms based on LLMs, such as the newly introduced custom GPTs by OpenAI. While custom GPTs provide various functionalities like web browsing and code execution, they also introduce significant security threats. In this paper, we conduct a comprehensive analysis of the security and privacy issues arising from the custom GPT platform. Our systematic examination categorizes potential attack scenarios into three threat models based on the role of the malicious actor, and identifies critical data exchange channels in custom GPTs. Utilizing the STRIDE threat modeling framework, we identify 26 potential attack vectors, with 19 being partially or fully validated in real-world settings. Our findings emphasize the urgent need for robust security and privacy measures in the custom GPT ecosystem, especially in light of the forthcoming launch of the official GPT store by OpenAI.

摘要: 大型语言模型(LLM)的出现极大地加速了各个领域广泛应用的开发。基于LLMS的专业平台建设有日益增长的趋势，例如OpenAI新推出的定制GPT。虽然自定义GPT提供了各种功能，如Web浏览和代码执行，但它们也带来了重大的安全威胁。在本文中，我们对定制GPT平台产生的安全和隐私问题进行了全面分析。我们的系统检查根据恶意行为者的角色将潜在的攻击场景分类为三种威胁模型，并确定了自定义GPT中的关键数据交换通道。利用STRIDE威胁建模框架，我们识别了26个潜在的攻击向量，其中19个在现实世界中得到了部分或完全的验证。我们的发现强调了定制GPT生态系统中强大的安全和隐私措施的迫切需要，特别是考虑到OpenAI即将推出官方GPT商店。



## **27. Identifying and Mitigating the Security Risks of Generative AI**

识别和缓解生成性人工智能的安全风险 cs.AI

**SubmitDate**: 2023-12-29    [abs](http://arxiv.org/abs/2308.14840v4) [paper-pdf](http://arxiv.org/pdf/2308.14840v4)

**Authors**: Clark Barrett, Brad Boyd, Elie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.

摘要: 每一项重大技术发明都会重新面临两难境地--新技术既有可能被用来做好事，也有可能被用来做坏事。生成性人工智能(GenAI)技术，如大型语言模型(LLMS)和扩散模型，已经显示出非凡的能力(例如，上下文学习、代码完成以及文本到图像的生成和编辑)。然而，攻击者也可以利用GenAI来生成新的攻击，并提高现有攻击的速度和效率。本文报告了在谷歌(由斯坦福大学和威斯康星大学麦迪逊分校联合举办)举行的关于GenAI造成的两用困境的研讨会的结果。这篇论文并不是要全面的，而是试图综合研讨会的一些有趣的发现。我们就这一主题讨论社区的短期和长期目标。我们希望这篇论文既为讨论这一重要主题提供了一个起点，也为研究界可以努力解决的有趣问题提供了一个起点。



## **28. Task Contamination: Language Models May Not Be Few-Shot Anymore**

任务污染：语言模型可能不再少见 cs.CL

Accepted by AAAI 2024

**SubmitDate**: 2023-12-26    [abs](http://arxiv.org/abs/2312.16337v1) [paper-pdf](http://arxiv.org/pdf/2312.16337v1)

**Authors**: Changmao Li, Jeffrey Flanigan

**Abstract**: Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot and few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that on datasets released before the LLM training data creation date, LLMs perform surprisingly better than on datasets released after. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets released prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, task example extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for classification tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.

摘要: 大型语言模型(LLM)在各种零射击和少射击任务中提供了令人印象深刻的性能。然而，他们在零射击和少射击设置中的成功可能会受到任务污染的影响，这是一个尚未彻底检查的潜在限制。本文研究了LLMS的零炮和少炮性能是如何随时间发生变化的。利用GPT-3系列模型和其他几个最近开源的LLM，并控制数据集的难度，我们发现，在LLM训练数据创建日期之前发布的数据集上，LLM的性能出人意料地好于之后发布的数据集。这有力地表明，对于许多LLMS来说，在LLMS的训练数据创建日期之前发布的数据集的零激发和少激发评估存在任务污染。此外，我们利用训练数据检查、任务实例提取和成员关系推理攻击，揭示了任务污染的进一步证据。重要的是，我们发现，对于没有任务污染可能性的分类任务，LLMS在零和极少的情况下，很少表现出比简单多数基线有统计上的显著改善。



## **29. Vulnerability of Machine Learning Approaches Applied in IoT-based Smart Grid: A Review**

基于物联网的智能电网中机器学习方法的脆弱性：综述 cs.CR

**SubmitDate**: 2023-12-25    [abs](http://arxiv.org/abs/2308.15736v3) [paper-pdf](http://arxiv.org/pdf/2308.15736v3)

**Authors**: Zhenyong Zhang, Mengxiang Liu, Mingyang Sun, Ruilong Deng, Peng Cheng, Dusit Niyato, Mo-Yuen Chow, Jiming Chen

**Abstract**: Machine learning (ML) sees an increasing prevalence of being used in the internet-of-things (IoT)-based smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. We first highlight the specifics for constructing the adversarial attacks on MLsgAPPs. Then, the vulnerability of MLsgAPP is analyzed from both the aspects of the power system and ML model. Afterward, a comprehensive survey is conducted to review and compare existing studies about the adversarial attacks on MLsgAPPs in scenarios of generation, transmission, distribution, and consumption, and the countermeasures are reviewed according to the attacks that they defend against. Finally, the future research directions are discussed on the attacker's and defender's side, respectively. We also analyze the potential vulnerability of large language model-based (e.g., ChatGPT) power system applications. Overall, we encourage more researchers to contribute to investigating the adversarial issues of MLsgAPPs.

摘要: 机器学习(ML)在基于物联网(IoT)的智能电网中的应用越来越普遍。然而，ML的可信性是一个必须解决的严重问题，以适应基于ML的智能电网应用(MLsgAPP)的趋势。注入到电源信号中的对抗性失真将极大地影响系统的正常控制和运行。因此，对应用于安全关键电力系统背景下的MLsgAPP进行脆弱性评估势在必行。在本文中，我们提供了一个全面的进展，设计攻击和防御方法的MLsgAPP。与传统的ML安全研究不同，本文首次针对电力系统的特点对MLsgAPP的安全问题进行了综述。我们首先强调构造对MLsgAPP的对抗性攻击的细节。然后，从电力系统和ML模型两个方面分析了MLsgAPP的脆弱性。然后，对已有的针对MLsgAPP的生成、传输、分发、消费等场景下的对抗性攻击的研究进行了全面的回顾和比较，并根据它们所防御的攻击回顾了相应的对策。最后，分别从攻击方和防御方的角度讨论了今后的研究方向。我们还分析了基于大型语言模型(如ChatGPT)的电力系统应用的潜在脆弱性。总体而言，我们鼓励更多的研究人员为研究MLsgAPP的对抗性问题做出贡献。



## **30. From Shortcuts to Triggers: Backdoor Defense with Denoised PoE**

从快捷方式到触发器：利用去噪PoE进行后门防御 cs.CL

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2305.14910v2) [paper-pdf](http://arxiv.org/pdf/2305.14910v2)

**Authors**: Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen

**Abstract**: Language models are often at risk of diverse backdoor attacks, especially data poisoning. Thus, it is important to investigate defense solutions for addressing them. Existing backdoor defense methods mainly focus on backdoor attacks with explicit triggers, leaving a universal defense against various backdoor attacks with diverse triggers largely unexplored. In this paper, we propose an end-to-end ensemble-based backdoor defense framework, DPoE (Denoised Product-of-Experts), which is inspired by the shortcut nature of backdoor attacks, to defend various backdoor attacks. DPoE consists of two models: a shallow model that captures the backdoor shortcuts and a main model that is prevented from learning the backdoor shortcuts. To address the label flip caused by backdoor attackers, DPoE incorporates a denoising design. Experiments on SST-2 dataset show that DPoE significantly improves the defense performance against various types of backdoor triggers including word-level, sentence-level, and syntactic triggers. Furthermore, DPoE is also effective under a more challenging but practical setting that mixes multiple types of trigger.

摘要: 语言模型经常面临各种后门攻击的风险，尤其是数据中毒。因此，研究解决这些问题的防御解决方案非常重要。现有的后门防御方法主要集中在具有显式触发的后门攻击上，对于各种触发方式多样的后门攻击的通用防御在很大程度上还没有被探索。本文从后门攻击的捷径特性出发，提出了一种基于端到端集成的后门防御框架DPoE(去噪专家积)来防御各种后门攻击。DPoE由两个模型组成：一个是捕获后门快捷方式的浅层模型，另一个是防止学习后门快捷方式的主模型。为了解决后门攻击者造成的标签翻转问题，DPoE采用了降噪设计。在SST-2数据集上的实验表明，DPoE显著提高了对各种后门触发器的防御性能，包括单词级、句子级和句法级触发器。此外，DPoE在更具挑战性但实用的混合多种触发器的环境下也是有效的。



## **31. A Mutation-Based Method for Multi-Modal Jailbreaking Attack Detection**

一种基于变异的多模式越狱攻击检测方法 cs.CR

12 pages, 8 figures

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.10766v2) [paper-pdf](http://arxiv.org/pdf/2312.10766v2)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Xiaofei Xie, Yang Liu, Chao Shen

**Abstract**: Large Language Models and Multi-Modal LLMs have become pervasive, and so does the importance of their security; yet, modern LLMs are known to be vulnerable to jailbreaking attacks. These attacks can allow malicious users to exploit the models, making the case for effective jailbreak detection mechanisms an essential aspect of maintaining the integrity and trustworthiness of LLM-based applications. However, existing detection works on jailbreak attacks have limitations. Existing post-query-based strategies require target domain knowledge, and pre-query-based methods mainly focus on text-level attacks and fail to meet the increasingly complex multi-modal security requirements placed upon contemporary LLMs. This gap underscores the need for a more comprehensive approach to safeguarding these influential systems.   In this work, we propose JailGuard, the first mutation-based jailbreaking detection framework which supports both image and text modalities. Our key observation is that attack queries inherently possess less robustness compared to benign queries. Specifically, to confuse the model, attack queries are usually crafted with well-designed templates or complicate perturbations, leading to a fact that a slight disturbance in input may result in a drastic change in the response. This lack of robustness can be utilized in attack detection. Based on this intuition, we designed and implemented a detection framework comprising 19 different mutators and a divergence-based detection formula. To fully understand the effectiveness of our framework, we built the first multi-modal LLM jailbreaking attack dataset, which has 304 items of data, covering ten types of known jailbreaking attacks on image and text modalities. The evaluation suggests that JailGuard achieves the best detection accuracy of 89.38%/85.42% on image and text inputs, outperforming state-of-the-art defense methods by 15.28%.

摘要: 大型语言模型和多模式LLM已经变得无处不在，它们的安全性也变得非常重要；然而，众所周知，现代LLM容易受到越狱攻击。这些攻击允许恶意用户利用这些模型，使有效的越狱检测机制成为维护基于LLM的应用程序的完整性和可信性的重要方面。然而，现有的越狱攻击检测工作存在局限性。现有的基于查询后的策略需要目标领域的知识，而基于查询前的方法主要关注文本级别的攻击，不能满足当代LLMS日益复杂的多模式安全需求。这一差距突出表明，需要采取更全面的方法来保护这些有影响力的制度。在这项工作中，我们提出了第一个基于突变的越狱检测框架JailGuard，它同时支持图像和文本两种模式。我们的主要观察结果是，与良性查询相比，攻击查询固有的健壮性较差。具体地说，为了混淆模型，攻击查询通常是使用精心设计的模板或复杂的扰动来制作的，导致输入中的轻微干扰可能会导致响应的剧烈变化。这种健壮性的缺乏可用于攻击检测。基于这一直觉，我们设计并实现了一个由19个不同的突变子和基于散度的检测公式组成的检测框架。为了充分理解我们框架的有效性，我们构建了第一个多模式LLM越狱攻击数据集，该数据集包含304项数据，涵盖了针对图像和文本模式的十种已知越狱攻击类型。评估表明，JailGuard对图像和文本输入的检测准确率最高，达到89.38%/85.42%，比最先进的防御方法高出15.28%。



## **32. A Survey on Large Language Models for Software Engineering**

面向软件工程的大型语言模型综述 cs.SE

**SubmitDate**: 2023-12-23    [abs](http://arxiv.org/abs/2312.15223v1) [paper-pdf](http://arxiv.org/pdf/2312.15223v1)

**Authors**: Quanjun Zhang, Chunrong Fang, Yang Xie, Yaxin Zhang, Yun Yang, Weisong Sun, Shengcheng Yu, Zhenyu Chen

**Abstract**: Software Engineering (SE) is the systematic design, development, and maintenance of software applications, underpinning the digital infrastructure of our modern mainworld. Very recently, the SE community has seen a rapidly increasing number of techniques employing Large Language Models (LLMs) to automate a broad range of SE tasks. Nevertheless, existing information of the applications, effects, and possible limitations of LLMs within SE is still not well-studied.   In this paper, we provide a systematic survey to summarize the current state-of-the-art research in the LLM-based SE community. We summarize 30 representative LLMs of Source Code across three model architectures, 15 pre-training objectives across four categories, and 16 downstream tasks across five categories. We then present a detailed summarization of the recent SE studies for which LLMs are commonly utilized, including 155 studies for 43 specific code-related tasks across four crucial phases within the SE workflow. Besides, we summarize existing attempts to empirically evaluate LLMs in SE, such as benchmarks, empirical studies, and exploration of SE education. We also discuss several critical aspects of optimization and applications of LLMs in SE, such as security attacks, model tuning, and model compression. Finally, we highlight several challenges and potential opportunities on applying LLMs for future SE studies, such as exploring domain LLMs and constructing clean evaluation datasets. Overall, our work can help researchers gain a comprehensive understanding about the achievements of the existing LLM-based SE studies and promote the practical application of these techniques. Our artifacts are publicly available and will continuously updated at the living repository: \url{https://github.com/iSEngLab/AwesomeLLM4SE}.

摘要: 软件工程(SE)是软件应用程序的系统设计、开发和维护，是现代主流世界的数字基础设施的基础。最近，SE社区看到越来越多的技术使用大型语言模型(LLM)来自动化广泛的SE任务。然而，现有的关于低密度脂蛋白在SE中的应用、效果和可能的局限性的信息仍然没有得到很好的研究。在本文中，我们提供了一个系统的综述，以总结当前在基于LLM的SE社区的最新研究。我们总结了三个模型体系结构中具有代表性的30个源代码LLM，四个类别中的15个预培训目标，以及五个类别中的16个下游任务。然后，我们详细总结了最近经常使用LLM的SE研究，包括针对SE工作流程中四个关键阶段的43个特定代码相关任务的155个研究。此外，我们还总结了已有的对SE中的LLMS进行经验性评估的尝试，如基准、实证研究和SE教育探索。我们还讨论了LLMS在SE中优化和应用的几个关键方面，如安全攻击、模型调整和模型压缩。最后，我们强调了在未来的SE研究中应用LLMS的几个挑战和潜在的机会，例如探索领域LLMS和构建干净的评估数据集。总体而言，我们的工作可以帮助研究人员全面了解现有基于LLM的SE研究的成果，并促进这些技术的实际应用。我们的手工艺品是公开可用的，并将在实时存储库中不断更新：\url{https://github.com/iSEngLab/AwesomeLLM4SE}.



## **33. Spear Phishing With Large Language Models**

使用大型语言模型的鱼叉式网络钓鱼 cs.CY

16 pages, 10 figures

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2305.06972v3) [paper-pdf](http://arxiv.org/pdf/2305.06972v3)

**Authors**: Julian Hazell

**Abstract**: Recent progress in artificial intelligence (AI), particularly in the domain of large language models (LLMs), has resulted in powerful and versatile dual-use systems. This intelligence can be put towards a wide variety of beneficial tasks, yet it can also be used to cause harm. This study explores one such harm by examining how LLMs can be used for spear phishing, a form of cybercrime that involves manipulating targets into divulging sensitive information. I first explore LLMs' ability to assist with the reconnaissance and message generation stages of a spear phishing attack, where I find that LLMs are capable of assisting with the email generation phase of a spear phishing attack. To explore how LLMs could potentially be harnessed to scale spear phishing campaigns, I then create unique spear phishing messages for over 600 British Members of Parliament using OpenAI's GPT-3.5 and GPT-4 models. My findings provide some evidence that these messages are not only realistic but also cost-effective, with each email costing only a fraction of a cent to generate. Next, I demonstrate how basic prompt engineering can circumvent safeguards installed in LLMs, highlighting the need for further research into robust interventions that can help prevent models from being misused. To further address these evolving risks, I explore two potential solutions: structured access schemes, such as application programming interfaces, and LLM-based defensive systems.

摘要: 人工智能(AI)领域的最新进展，特别是在大型语言模型(LLMS)领域的进展，导致了强大而通用的两用系统。这种智慧可以用于各种各样有益的任务，但也可以用来造成伤害。这项研究通过研究LLMS如何被用于鱼叉式网络钓鱼来探索这样的危害，鱼叉式网络钓鱼是一种网络犯罪形式，涉及操纵目标泄露敏感信息。我首先探讨LLMS协助鱼叉式网络钓鱼攻击的侦察和消息生成阶段的能力，我发现LLMS能够协助鱼叉式网络钓鱼攻击的电子邮件生成阶段。为了探索如何潜在地利用LLMS来扩大鱼叉式网络钓鱼活动，我随后使用OpenAI的GPT-3.5和GPT-4模型为600多名英国国会议员创建了独特的鱼叉式网络钓鱼消息。我的发现提供了一些证据，证明这些信息不仅现实，而且具有成本效益，每封电子邮件的生成成本只有一分钱的零头。接下来，我将演示基本的即时工程如何绕过安装在低成本管理系统中的保障措施，强调有必要对能够帮助防止模型被滥用的强大干预措施进行进一步研究。为了进一步应对这些不断变化的风险，我探索了两个潜在的解决方案：结构化访问方案，如应用程序编程接口，以及基于LLM的防御系统。



## **34. MetaAID 2.5: A Secure Framework for Developing Metaverse Applications via Large Language Models**

MetaAID2.5：一个通过大型语言模型开发Metaverse应用程序的安全框架 cs.CR

**SubmitDate**: 2023-12-22    [abs](http://arxiv.org/abs/2312.14480v1) [paper-pdf](http://arxiv.org/pdf/2312.14480v1)

**Authors**: Hongyin Zhu

**Abstract**: Large language models (LLMs) are increasingly being used in Metaverse environments to generate dynamic and realistic content and to control the behavior of non-player characters (NPCs). However, the cybersecurity concerns associated with LLMs have become increasingly prominent. Previous research has primarily focused on patching system vulnerabilities to enhance cybersecurity, but these approaches are not well-suited to the Metaverse, where the virtual space is more complex, LLMs are vulnerable, and ethical user interaction is critical. Moreover, the scope of cybersecurity in the Metaverse is expected to expand significantly. This paper proposes a method for enhancing cybersecurity through the simulation of user interaction with LLMs. Our goal is to educate users and strengthen their defense capabilities through exposure to a comprehensive simulation system. This system includes extensive Metaverse cybersecurity Q&A and attack simulation scenarios. By engaging with these, users will improve their ability to recognize and withstand risks. Additionally, to address the ethical implications of user input, we propose using LLMs as evaluators to assess user content across five dimensions. We further adapt the models through vocabulary expansion training to better understand personalized inputs and emoticons. We conduct experiments on multiple LLMs and find that our approach is effective.

摘要: 大型语言模型（LLM）越来越多地用于Metaverse环境中，以生成动态和逼真的内容，并控制非玩家角色（NPC）的行为。然而，与LLM相关的网络安全问题变得越来越突出。以前的研究主要集中在修补系统漏洞以增强网络安全，但这些方法并不适合Metaverse，其中虚拟空间更复杂，LLM易受攻击，道德用户交互至关重要。此外，Metaverse的网络安全范围预计将大幅扩大。本文提出了一种通过模拟用户与LLM的交互来增强网络安全的方法。我们的目标是通过全面的模拟系统教育用户并加强他们的防御能力。该系统包括广泛的Metaverse网络安全问答和攻击模拟场景。通过参与这些活动，用户将提高识别和抵御风险的能力。此外，为了解决用户输入的道德影响，我们建议使用LLM作为评估者，以评估用户内容的五个维度。我们通过词汇扩展训练进一步调整模型，以更好地理解个性化输入和表情符号。我们在多个LLM上进行了实验，发现我们的方法是有效的。



## **35. HW-V2W-Map: Hardware Vulnerability to Weakness Mapping Framework for Root Cause Analysis with GPT-assisted Mitigation Suggestion**

HW-V2W-Map：GPT辅助缓解建议的根本原因分析的硬件弱点映射框架 cs.CR

22 pages, 10 pages appendix, 10 figures, Submitted to ACM TODAES

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.13530v1) [paper-pdf](http://arxiv.org/pdf/2312.13530v1)

**Authors**: Yu-Zheng Lin, Muntasir Mamun, Muhtasim Alam Chowdhury, Shuyu Cai, Mingyu Zhu, Banafsheh Saber Latibari, Kevin Immanuel Gubbi, Najmeh Nazari Bavarsad, Arjun Caputo, Avesta Sasan, Houman Homayoun, Setareh Rafatirad, Pratik Satam, Soheil Salehi

**Abstract**: The escalating complexity of modern computing frameworks has resulted in a surge in the cybersecurity vulnerabilities reported to the National Vulnerability Database (NVD) by practitioners. Despite the fact that the stature of NVD is one of the most significant databases for the latest insights into vulnerabilities, extracting meaningful trends from such a large amount of unstructured data is still challenging without the application of suitable technological methodologies. Previous efforts have mostly concentrated on software vulnerabilities; however, a holistic strategy incorporates approaches for mitigating vulnerabilities, score prediction, and a knowledge-generating system that may extract relevant insights from the Common Weakness Enumeration (CWE) and Common Vulnerability Exchange (CVE) databases is notably absent. As the number of hardware attacks on Internet of Things (IoT) devices continues to rapidly increase, we present the Hardware Vulnerability to Weakness Mapping (HW-V2W-Map) Framework, which is a Machine Learning (ML) framework focusing on hardware vulnerabilities and IoT security. The architecture that we have proposed incorporates an Ontology-driven Storytelling framework, which automates the process of updating the ontology in order to recognize patterns and evolution of vulnerabilities over time and provides approaches for mitigating the vulnerabilities. The repercussions of vulnerabilities can be mitigated as a result of this, and conversely, future exposures can be predicted and prevented. Furthermore, our proposed framework utilized Generative Pre-trained Transformer (GPT) Large Language Models (LLMs) to provide mitigation suggestions.

摘要: 现代计算框架的日益复杂导致从业者向国家漏洞数据库(NVD)报告的网络安全漏洞激增。尽管NVD的地位是最新洞察漏洞的最重要的数据库之一，但如果没有适当的技术方法的应用，从如此大量的非结构化数据中提取有意义的趋势仍然是具有挑战性的。以前的工作主要集中在软件漏洞上；然而，整体战略结合了缓解漏洞的方法、分数预测，并且明显缺乏可以从共同弱点枚举(CWE)和共同漏洞交换(CVE)数据库中提取相关见解的知识生成系统。针对物联网(IoT)设备遭受硬件攻击的情况，提出了硬件弱点映射(HW-V2W-Map)框架，这是一个关注硬件脆弱性和物联网安全的机器学习(ML)框架。我们建议的体系结构结合了本体驱动的故事讲述框架，该框架自动更新本体的过程，以便识别漏洞的模式和随时间的演变，并提供缓解漏洞的方法。因此，可以减轻漏洞的影响，反过来，可以预测和预防未来的风险暴露。此外，我们提出的框架利用产生式预训练转换器(GPT)大型语言模型(LLMS)来提供缓解建议。



## **36. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2023-12-21    [abs](http://arxiv.org/abs/2312.14197v1) [paper-pdf](http://arxiv.org/pdf/2312.14197v1)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: Recent remarkable advancements in large language models (LLMs) have led to their widespread adoption in various applications. A key feature of these applications is the combination of LLMs with external content, where user instructions and third-party content are combined to create prompts for LLM processing. These applications, however, are vulnerable to indirect prompt injection attacks, where malicious instructions embedded within external content compromise LLM's output, causing their responses to deviate from user expectations. Despite the discovery of this security issue, no comprehensive analysis of indirect prompt injection attacks on different LLMs is available due to the lack of a benchmark. Furthermore, no effective defense has been proposed.   In this work, we introduce the first benchmark, BIPIA, to measure the robustness of various LLMs and defenses against indirect prompt injection attacks. Our experiments reveal that LLMs with greater capabilities exhibit more vulnerable to indirect prompt injection attacks for text tasks, resulting in a higher ASR. We hypothesize that indirect prompt injection attacks are mainly due to the LLMs' inability to distinguish between instructions and external content. Based on this conjecture, we propose four black-box methods based on prompt learning and a white-box defense methods based on fine-tuning with adversarial training to enable LLMs to distinguish between instructions and external content and ignore instructions in the external content. Our experimental results show that our black-box defense methods can effectively reduce ASR but cannot completely thwart indirect prompt injection attacks, while our white-box defense method can reduce ASR to nearly zero with little adverse impact on the LLM's performance on general tasks. We hope that our benchmark and defenses can inspire future work in this important area.

摘要: 最近，大型语言模型(LLM)的显著进步导致了它们在各种应用中的广泛采用。这些应用程序的一个关键功能是将LLM与外部内容相结合，其中结合了用户指令和第三方内容，以创建LLM处理的提示。然而，这些应用程序容易受到间接提示注入攻击，在这种攻击中，嵌入在外部内容中的恶意指令会危及LLM的输出，导致它们的响应偏离用户预期。尽管发现了这个安全问题，但由于缺乏基准，对不同LLM的间接即时注入攻击没有全面的分析。此外，还没有提出有效的防御措施。在这项工作中，我们引入了第一个基准，BIPIA，来衡量各种LLM的健壮性和对间接即时注入攻击的防御。我们的实验表明，具有更大能力的LLM更容易受到文本任务的间接提示注入攻击，从而导致更高的ASR。我们假设间接提示注入攻击主要是由于LLMS无法区分指令和外部内容。基于这一猜想，我们提出了四种基于快速学习的黑盒方法和一种基于微调和对抗性训练的白盒防御方法，使LLMS能够区分指令和外部内容，并忽略外部内容中的指令。我们的实验结果表明，我们的黑盒防御方法可以有效地降低ASR，但不能完全阻止间接提示注入攻击，而我们的白盒防御方法可以将ASR降低到几乎为零，并且对LLM在一般任务上的性能影响很小。我们希望我们的基准和辩护能够激励这一重要领域的未来工作。



## **37. Universal and Transferable Adversarial Attacks on Aligned Language Models**

对对齐语言模型的通用和可转移的对抗性攻击 cs.CL

Website: http://llm-attacks.org/

**SubmitDate**: 2023-12-20    [abs](http://arxiv.org/abs/2307.15043v2) [paper-pdf](http://arxiv.org/pdf/2307.15043v2)

**Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

摘要: 由于“开箱即用”的大型语言模型能够生成大量令人反感的内容，最近的工作重点是调整这些模型，以试图防止不必要的生成。虽然在规避这些措施方面取得了一些成功--即所谓的针对LLMS的“越狱”--但这些攻击需要大量的人类智慧，而且在实践中是脆弱的。在本文中，我们提出了一种简单有效的攻击方法，使对齐的语言模型产生令人反感的行为。具体地说，我们的方法找到了一个后缀，当附加到LLM的广泛查询中以产生令人反感的内容时，旨在最大化该模型产生肯定响应(而不是拒绝回答)的概率。然而，我们的方法不依赖于人工设计，而是通过贪婪和基于梯度的搜索技术相结合来自动生成这些对抗性后缀，并且改进了过去的自动提示生成方法。令人惊讶的是，我们发现我们的方法生成的对抗性提示是相当可转移的，包括到黑盒，公开发布的LLM。具体地说，我们对多个提示(即，要求许多不同类型的不良内容的查询)以及多个模型(在我们的案例中，Vicuna-7B和13B)训练对抗性攻击后缀。这样做时，生成的攻击后缀能够在ChatGPT、Bard和Claude的公共接口以及开源LLM(如llama-2-chat、Pythia、Falcon和其他)中诱导令人反感的内容。总而言之，这项工作极大地推进了针对对齐语言模型的对抗性攻击的最新水平，提出了如何防止此类系统产生令人反感的信息的重要问题。代码可在githorb.com/llm-Attages/llm-Attack上找到。



## **38. Robust Contrastive Language-Image Pre-training against Data Poisoning and Backdoor Attacks**

抗数据中毒和后门攻击的鲁棒对比图像预训练 cs.CV

**SubmitDate**: 2023-12-19    [abs](http://arxiv.org/abs/2303.06854v2) [paper-pdf](http://arxiv.org/pdf/2303.06854v2)

**Authors**: Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose ROCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. ROCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that ROCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, ROCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, ROCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.

摘要: 对比视觉语言表征学习通过从互联网上抓取的数百万个图像-标题对中学习，实现了最先进的零镜头分类性能。然而，为CLIP等大型多模态模型提供动力的大量数据使它们极易受到各种类型的有针对性的数据中毒和后门攻击。尽管存在这种脆弱性，但针对此类攻击的强大对比视觉语言预训练仍然没有得到解决。在这项工作中，我们提出了ROCLIP，这是第一个有效的方法，用于鲁棒的预训练多模态视觉语言模型，以对抗有针对性的数据中毒和后门攻击。ROCLIP通过考虑一个相对较大且变化的随机字幕池，并每隔几个epoch将每个图像与池中最相似的文本（而不是其自身的字幕）进行匹配，有效地打破了中毒图像-字幕对之间的关联。它还利用图像和文本增强来进一步加强防御并提高模型的性能。我们广泛的实验表明，ROCLIP在预训练CLIP模型期间使最先进的有针对性的数据中毒和后门攻击无效。特别是，ROCLIP将目标数据中毒攻击的成功率从93.75%降低到12.5%，后门攻击的成功率降低到0%，同时将模型的线性探测性能提高了10%，并保持了与CLIP相似的零射击性能。通过增加匹配频率，ROCLIP能够防御强攻击，这些攻击将向数据添加高达1%的中毒示例，并成功保持12.5%的低攻击成功率，同时在某些任务上牺牲性能。



## **39. PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**

PoisonPrompt：对基于提示的大型语言模型的后门攻击 cs.CL

To Appear in IEEE ICASSP 2024, code is available at:  https://github.com/grasses/PoisonPrompt

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2310.12439v2) [paper-pdf](http://arxiv.org/pdf/2310.12439v2)

**Authors**: Hongwei Yao, Jian Lou, Zhan Qin

**Abstract**: Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.

摘要: 最近，机器人显著提高了预训练的大型语言模型（LLM）在各种下游任务上的性能，使它们在各种LLM应用场景中变得越来越不可或缺。然而，后门漏洞，一个严重的安全威胁，可以恶意改变受害者模型的正常预测，还没有充分探讨基于恶意的LLM。在本文中，我们提出了POISONPROMPT，一种新的后门攻击能够成功地妥协的硬和软基于木马的LLM。我们评估的有效性，保真度和鲁棒性的POISONPROMPT通过广泛的实验，三个流行的提示方法，使用六个数据集和三个广泛使用的LLM。我们的研究结果强调了后门攻击对基于Linux的LLM构成的潜在安全威胁，并强调了在这一领域进一步研究的必要性。



## **40. A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models**

大型语言模型中的攻击技术、实现和防御策略综述 cs.CR

Accepted to be published in the Proceedings of the 3rd International  Conference on Ubiquitous Security 2023 (UbiSec-2023)

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.10982v1) [paper-pdf](http://arxiv.org/pdf/2312.10982v1)

**Authors**: Aysan Esmradi, Daniel Wankit Yip, Chun Fai Chan

**Abstract**: Ensuring the security of large language models (LLMs) is an ongoing challenge despite their widespread popularity. Developers work to enhance LLMs security, but vulnerabilities persist, even in advanced versions like GPT-4. Attackers exploit these weaknesses, highlighting the need for proactive cybersecurity measures in AI model development. This article explores two attack categories: attacks on models themselves and attacks on model applications. The former requires expertise, access to model data, and significant implementation time, while the latter is more accessible to attackers and has seen increased attention. Our study reviews over 100 recent research works, providing an in-depth analysis of each attack type. We identify the latest attack methods and explore various approaches to carry them out. We thoroughly investigate mitigation techniques, assessing their effectiveness and limitations. Furthermore, we summarize future defenses against these attacks. We also examine real-world techniques, including reported and our implemented attacks on LLMs, to consolidate our findings. Our research highlights the urgency of addressing security concerns and aims to enhance the understanding of LLM attacks, contributing to robust defense development in this evolving domain.

摘要: 尽管大型语言模型(LLM)广受欢迎，但确保其安全性仍是一个持续的挑战。开发人员努力增强LLMS的安全性，但漏洞仍然存在，即使是在GPT-4这样的高级版本中也是如此。攻击者利用这些弱点，突显了在人工智能模型开发中采取主动网络安全措施的必要性。本文探讨了两种攻击类别：对模型本身的攻击和对模型应用程序的攻击。前者需要专业知识、对模型数据的访问和大量的实施时间，而后者更容易被攻击者访问，并受到越来越多的关注。我们的研究回顾了100多项最新的研究工作，对每种攻击类型进行了深入的分析。我们确定了最新的攻击方法，并探索了执行它们的各种方法。我们深入研究缓解技术，评估其有效性和局限性。此外，我们还总结了未来针对这些攻击的防御措施。我们还检查了现实世界的技术，包括报告的和我们实施的对LLM的攻击，以巩固我们的发现。我们的研究强调了解决安全问题的紧迫性，并旨在加强对LLM攻击的理解，为这个不断发展的领域中强有力的防御发展做出贡献。



## **41. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

No-Skim：基于略读的语言模型的效率稳健性评价 cs.CR

**SubmitDate**: 2023-12-18    [abs](http://arxiv.org/abs/2312.09494v2) [paper-pdf](http://arxiv.org/pdf/2312.09494v2)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

摘要: 为了降低大型语言模型(LLM)的计算代价和能量消耗，基于略读的加速算法在保留语义重要的标记的同时，动态地沿着LLM的层次逐步丢弃输入序列中不重要的标记。然而，我们的工作首次揭示了加速可能容易受到拒绝服务(DoS)攻击。在本文中，我们提出了一个通用的框架No-Skim，以帮助基于略读的LLM的所有者理解和度量其加速方案的健壮性。具体地说，我们的框架在字符级和令牌级搜索最小和不可察觉的扰动，以生成足以增加剩余令牌率的对抗性输入，从而增加计算成本和能量消耗。我们在GLUE基准上系统地评估了包括Bert和Roberta在内的各种LLM架构中掠读加速的脆弱性。在最坏的情况下，No-Skim发现的扰动大大增加了LLM的运行成本，平均超过145%。此外，No-Skim将评估框架扩展到各种场景，使评估可以在不同的知识水平下进行。



## **42. Privacy-Aware Document Visual Question Answering**

隐私感知文档视觉问答 cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.10108v1) [paper-pdf](http://arxiv.org/pdf/2312.10108v1)

**Authors**: Rubèn Tito, Khanh Nguyen, Marlon Tobaben, Raouf Kerkouche, Mohamed Ali Souibgui, Kangsoo Jung, Lei Kang, Ernest Valveny, Antti Honkela, Mario Fritz, Dimosthenis Karatzas

**Abstract**: Document Visual Question Answering (DocVQA) is a fast growing branch of document understanding. Despite the fact that documents contain sensitive or copyrighted information, none of the current DocVQA methods offers strong privacy guarantees.   In this work, we explore privacy in the domain of DocVQA for the first time. We highlight privacy issues in state of the art multi-modal LLM models used for DocVQA, and explore possible solutions.   Specifically, we focus on the invoice processing use case as a realistic, widely used scenario for document understanding, and propose a large scale DocVQA dataset comprising invoice documents and associated questions and answers. We employ a federated learning scheme, that reflects the real-life distribution of documents in different businesses, and we explore the use case where the ID of the invoice issuer is the sensitive information to be protected.   We demonstrate that non-private models tend to memorise, behaviour that can lead to exposing private information. We then evaluate baseline training schemes employing federated learning and differential privacy in this multi-modal scenario, where the sensitive information might be exposed through any of the two input modalities: vision (document image) or language (OCR tokens).   Finally, we design an attack exploiting the memorisation effect of the model, and demonstrate its effectiveness in probing different DocVQA models.

摘要: 文档视觉问答(DocVQA)是文档理解领域中发展迅速的一个分支。尽管文档包含敏感或受版权保护的信息，但当前的DocVQA方法都不能提供强有力的隐私保障。在这项工作中，我们首次探索了DocVQA领域的隐私。我们重点介绍了用于DocVQA的最先进的多模式LLM模型中的隐私问题，并探索了可能的解决方案。具体地说，我们将发票处理用例作为一个现实的、广泛使用的文档理解场景来关注，并提出了一个大规模的DocVQA数据集，包括发票文档和相关的问答。我们采用了联合学习方案，反映了文档在不同业务中的真实分布，并探索了发票开具人的ID是要保护的敏感信息的用例。我们证明，非私人模式往往会记忆，这是可能导致私人信息泄露的行为。然后，我们评估了在这种多模式场景中使用联合学习和差异隐私的基线训练方案，其中敏感信息可能通过两种输入通道中的任何一种暴露：视觉(文档图像)或语言(OCR令牌)。最后，我们设计了一个利用该模型的记忆效应的攻击，并在探测不同的DocVQA模型时展示了它的有效性。



## **43. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的可解释的基于梯度的对抗性攻击 cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，防御这些攻击是可能的：对抗性攻击生成无限但不可读的胡言乱语提示，可通过基于困惑的过滤器检测；手动越狱攻击创建可读的提示，但由于人类创造力的必要性，其数量有限，允许轻松阻止。在本文中，我们证明了这些解决方案可能过于乐观。我们介绍了AutoDAN，一种可解释的、基于梯度的对抗性攻击，它融合了这两种攻击类型的优点。在越狱和可读性双重目标的指导下，AutoDAN从左到右一个接一个地优化和生成令牌，产生可读的提示，绕过困惑过滤器，同时保持高攻击成功率。值得注意的是，这些使用渐变从零开始生成的提示是可解释的和多样化的，新出现的策略通常出现在手动越狱攻击中。当使用有限的训练数据或单一代理模型时，它们还概括到不可预见的有害行为，并比不可读的同行更好地转移到黑盒LLM。此外，我们通过使用定制目标自动泄漏系统提示来展示AutoDAN的多功能性。我们的工作为红色团队LLM提供了一种新的方法，并通过可解释性来理解越狱机制。



## **44. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

图：通过排版视觉提示越狱的大型视觉语言模型 cs.CR

Technical Report

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2311.05608v2) [paper-pdf](http://arxiv.org/pdf/2311.05608v2)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Ensuring the safety of artificial intelligence-generated content (AIGC) is a longstanding topic in the artificial intelligence (AI) community, and the safety concerns associated with Large Language Models (LLMs) have been widely investigated. Recently, large vision-language models (VLMs) represent an unprecedented revolution, as they are built upon LLMs but can incorporate additional modalities (e.g., images). However, the safety of VLMs lacks systematic evaluation, and there may be an overconfidence in the safety guarantees provided by their underlying LLMs. In this paper, to demonstrate that introducing additional modality modules leads to unforeseen AI safety issues, we propose FigStep, a straightforward yet effective jailbreaking algorithm against VLMs. Instead of feeding textual harmful instructions directly, FigStep converts the harmful content into images through typography to bypass the safety alignment within the textual module of the VLMs, inducing VLMs to output unsafe responses that violate common AI safety policies. In our evaluation, we manually review 46,500 model responses generated by 3 families of the promising open-source VLMs, i.e., LLaVA, MiniGPT4, and CogVLM (a total of 6 VLMs). The experimental results show that FigStep can achieve an average attack success rate of 82.50% on 500 harmful queries in 10 topics. Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages an OCR detector to filter harmful queries. Above all, our work reveals that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.

摘要: 确保人工智能生成内容(AIGC)的安全性是人工智能(AI)领域的一个长期话题，与大型语言模型(LLM)相关的安全问题已被广泛调查。最近，大型视觉语言模型(VLM)代表了一场前所未有的革命，因为它们建立在LLM之上，但可以结合其他形式(例如图像)。然而，超低层管理系统的安全性缺乏系统的评估，可能对其底层低层管理系统提供的安全保证过于自信。在本文中，为了证明引入额外的通道模块会导致不可预见的人工智能安全问题，我们提出了FigStep，一种针对VLM的简单而有效的越狱算法。FigStep没有直接提供文本有害指令，而是通过排版将有害内容转换为图像，以绕过VLM文本模块内的安全对齐，诱导VLM输出违反常见AI安全策略的不安全响应。在我们的评估中，我们手动审查了3个有前途的开源VLM家族，即LLaVA、MiniGPT4和CogVLM(总共6个VLM)生成的46,500个模型响应。实验结果表明，FigStep对10个主题500个有害查询的平均攻击成功率为82.50%。此外，我们还演示了FigStep的方法甚至可以越狱GPT-4V，它已经利用OCR检测器来过滤有害查询。最重要的是，我们的工作揭示了VLM容易受到越狱攻击，这突显了视觉和文本通道之间新的安全对齐的必要性。



## **45. Efficient Representation of the Activation Space in Deep Neural Networks**

深度神经网络中激活空间的有效表示 cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.

摘要: 深度神经网络(DNN)的激活空间表示被广泛用于自然语言处理、异常检测和语音识别等任务。由于这些任务的多样性和DNN的巨大规模，高效和独立于任务的激活表示变得至关重要。经验p值被用来量化与已知输入产生的激活相比，观察到的节点激活的相对强度。尽管如此，为这些计算保留原始数据会增加内存资源消耗，并引发隐私问题。为此，我们提出了一个与模型无关的框架，用于使用节点特定的直方图来创建DNN中的激活表示，以计算观察到的激活的p值，而不保留已知的输入。我们提出的方法在不同下游任务的多个网络架构上进行验证，并与核密度估计和蛮力经验基线进行比较，显示出良好的潜力。此外，该框架减少了30%的内存使用量，p值计算时间最多提高了4倍，同时在下游任务中保持了最先进的检测能力，例如检测对抗性攻击和合成内容。此外，由于我们不在推理时保留原始数据，因此我们可能会降低对攻击和隐私问题的易感性。



## **46. Causality Analysis for Evaluating the Security of Large Language Models**

大型语言模型安全性评估的因果分析 cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.

摘要: 大型语言模型(LLM)，如GPT和Llama2，在许多安全关键型应用中越来越多地被采用。因此，他们的安全至关重要。即使在从人类反馈中强化学习(RLHF)方面花费了大量的努力，最近的研究表明LLMS仍然受到诸如对抗性扰动和特洛伊木马攻击的攻击。因此，需要进一步研究，以评估其安全性和/或了解其缺乏安全性。在这项工作中，我们提出了一个框架，用于在标记、层和神经元水平上进行LLMS的轻量级因果分析。我们将我们的框架应用于开源LLM，如Llama2和Vicuna，并有多个有趣的发现。基于层级因果关系分析，我们发现RLHF具有对有害提示的模型过度拟合的效果。这意味着这种安全很容易被“不寻常的”有害提示所克服。作为证据，我们提出了一种对抗性扰动方法，在2023年木马检测大赛的红队任务上达到了100%的攻击成功率。此外，我们证明了在Llama2和Vicuna2中都存在一个神秘的神经元，它对输出具有不合理的高因果效应。虽然我们不确定为什么会有这样的神经元存在，但我们证明了有可能进行针对该特定神经元的“特洛伊木马”攻击，以完全削弱LLM，即我们可以为提示生成可转移的后缀，这些后缀经常使LLM产生无意义的响应。



## **47. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

DeceptPrompt：通过对抗性自然语言指令利用LLM驱动的代码生成 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

摘要: 随着大型语言模型(LLMS)的发展，在代码生成方面取得了重大进展，使LLMS能够将自然语言转换为编程代码。这些CodeLLM已被广大用户和组织广泛接受。然而，代码中隐藏着一个危险的性质，那就是存在致命的漏洞。虽然一些LLM提供商试图通过与人类的指导保持一致来解决这些问题，但这些努力并不能使Code LLM实用和健壮。如果不深入了解LLMS在实际最坏情况下的性能，将它们应用于各种现实世界应用将是令人担忧的。在这篇文章中，我们回答了一个关键问题：现有的代码LLM是否不会生成易受攻击的代码？如果不是，此问题在实际部署方案中可能的最大严重程度是多少？在本文中，我们介绍了DeceptPrompt算法，它可以生成敌意的自然语言指令，这些指令驱动Code LLMS生成有漏洞的功能正确的代码。DeceptPrompt是通过基于系统进化的算法实现的，具有细粒度的损耗设计。DeceptPrompt的独特优势使我们能够找到具有完全良性和非方向性语义的自然前缀/后缀，同时对诱使Code LLMS生成易受攻击的代码具有强大的能力。这一功能使我们能够在用户使用自然语言的真实场景中对这些LLM进行几乎最糟糕的红色团队。我们在DeceptPrompt上的大量实验和分析不仅验证了我们方法的有效性，而且揭示了LLMS在代码生成任务中的巨大弱点。当应用优化的前缀/后缀时，与不应用前缀/后缀相比，攻击成功率(ASR)将平均提高50%。



## **48. Maatphor: Automated Variant Analysis for Prompt Injection Attacks**

Maatphor：针对即时注入攻击的自动变量分析 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.11513v1) [paper-pdf](http://arxiv.org/pdf/2312.11513v1)

**Authors**: Ahmed Salem, Andrew Paverd, Boris Köpf

**Abstract**: Prompt injection has emerged as a serious security threat to large language models (LLMs). At present, the current best-practice for defending against newly-discovered prompt injection techniques is to add additional guardrails to the system (e.g., by updating the system prompt or using classifiers on the input and/or output of the model.) However, in the same way that variants of a piece of malware are created to evade anti-virus software, variants of a prompt injection can be created to evade the LLM's guardrails. Ideally, when a new prompt injection technique is discovered, candidate defenses should be tested not only against the successful prompt injection, but also against possible variants.   In this work, we present, a tool to assist defenders in performing automated variant analysis of known prompt injection attacks. This involves solving two main challenges: (1) automatically generating variants of a given prompt according, and (2) automatically determining whether a variant was effective based only on the output of the model. This tool can also assist in generating datasets for jailbreak and prompt injection attacks, thus overcoming the scarcity of data in this domain.   We evaluate Maatphor on three different types of prompt injection tasks. Starting from an ineffective (0%) seed prompt, Maatphor consistently generates variants that are at least 60% effective within the first 40 iterations.

摘要: 快速注入已成为大型语言模型(LLM)的严重安全威胁。目前，防御新发现的提示注入技术的最佳实践是向系统添加额外的护栏(例如，通过更新系统提示或使用关于模型的输入和/或输出的分类器)。然而，就像创建恶意软件的变体来逃避反病毒软件一样，也可以创建即时注入的变体来逃避LLM的护栏。理想情况下，当一种新的快速注射技术被发现时，候选防御不仅应该针对成功的快速注射进行测试，而且应该针对可能的变体进行测试。在这项工作中，我们提出了一个工具，以帮助防御者执行自动变异分析已知的即时注入攻击。这涉及解决两个主要挑战：(1)根据给定提示自动生成变体，以及(2)仅基于模型的输出自动确定变体是否有效。该工具还可以帮助生成越狱和提示注入攻击的数据集，从而克服该领域数据稀缺的问题。我们在三种不同类型的快速注射任务中对Maatphor进行了评估。从一个无效的(0%)种子提示开始，Maatphor始终生成在前40次迭代中至少60%有效的变体。



## **49. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

基于自提示校正的针对精调大型语言模型的实用隶属度推理攻击 cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 隶属度推断攻击（MIA）旨在推断目标数据记录是否已被用于模型训练。先前的尝试已经通过MIA量化了语言模型（LM）的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型（LLM）上导致显着的隐私泄露仍然没有达成共识。现有的针对LM的MIA攻击可以分为两类：无参考攻击和基于参考的攻击。它们都基于这样的假设，即训练记录始终具有较高的被采样概率。然而，这一假设严重依赖于目标模型的过拟合，这将通过多种正则化方法和LLM的泛化来缓解。基于参考的攻击似乎在LLM中取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来测量更可靠的隶属度信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总的来说，现有的MIA无法有效地揭露隐私泄漏，而实际的微调LLM是过度拟合和私人的。提出了一种基于自校准概率变差的成员推理攻击（SPV-MIA）。具体来说，由于LLM中的记忆在训练过程中是不可避免的，并且发生在过拟合之前，因此我们引入了一个更可靠的隶属度信号，概率变化，它基于记忆而不是过拟合。此外，我们引入了一种自提示方法，该方法通过提示目标LLM本身来构造数据集以微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **50. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全对齐：作为上下文攻击的弱对齐总结 cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型（LLM）的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否充分符合安全考虑？我们的研究专注于通过对抗性攻击获得的安全敏感文档，揭示了各种NLP任务的安全性差异。例如，LLM可以有效地总结恶意的长文档，但通常拒绝翻译它们。这种差异突出了一个以前未发现的漏洞：利用安全性较弱的任务（如摘要）的攻击可能会损害传统上被认为更强大的任务（如翻译和问答（QA））的完整性。此外，同时使用多个NLP任务，安全性较低，增加了LLM无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama 2模型和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



