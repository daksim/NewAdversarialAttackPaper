# Latest Large Language Model Attack Papers
**update at 2024-12-26 10:01:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can LLMs Obfuscate Code? A Systematic Analysis of Large Language Models into Assembly Code Obfuscation**

LLM可以混淆代码吗？大型语言模型到汇编代码混淆的系统分析 cs.CR

To appear in AAAI 2025, Main Track

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.16135v2) [paper-pdf](http://arxiv.org/pdf/2412.16135v2)

**Authors**: Seyedreza Mohseni, Seyedali Mohammadi, Deepa Tilwani, Yash Saxena, Gerald Ndawula, Sriram Vema, Edward Raff, Manas Gaur

**Abstract**: Malware authors often employ code obfuscations to make their malware harder to detect. Existing tools for generating obfuscated code often require access to the original source code (e.g., C++ or Java), and adding new obfuscations is a non-trivial, labor-intensive process. In this study, we ask the following question: Can Large Language Models (LLMs) potentially generate a new obfuscated assembly code? If so, this poses a risk to anti-virus engines and potentially increases the flexibility of attackers to create new obfuscation patterns. We answer this in the affirmative by developing the MetamorphASM benchmark comprising MetamorphASM Dataset (MAD) along with three code obfuscation techniques: dead code, register substitution, and control flow change. The MetamorphASM systematically evaluates the ability of LLMs to generate and analyze obfuscated code using MAD, which contains 328,200 obfuscated assembly code samples. We release this dataset and analyze the success rate of various LLMs (e.g., GPT-3.5/4, GPT-4o-mini, Starcoder, CodeGemma, CodeLlama, CodeT5, and LLaMA 3.1) in generating obfuscated assembly code. The evaluation was performed using established information-theoretic metrics and manual human review to ensure correctness and provide the foundation for researchers to study and develop remediations to this risk. The source code can be found at the following GitHub link: https://github.com/mohammadi-ali/MetamorphASM.

摘要: 恶意软件作者经常使用代码混淆来使他们的恶意软件更难被检测到。现有的用于生成混淆代码的工具通常需要访问原始源代码(例如，C++或Java)，而添加新的混淆并不是一个琐碎的、劳动密集型的过程。在这项研究中，我们问了以下问题：大型语言模型(LLM)是否有可能生成新的混淆汇编代码？如果是这样的话，这会给反病毒引擎带来风险，并可能增加攻击者创建新的混淆模式的灵活性。我们通过开发包含变形ASM数据集(MAD)以及三种代码混淆技术的变形ASM基准来肯定地回答这个问题：死代码、寄存器替换和控制流更改。变质ASM系统地评估LLMS使用MAD生成和分析混淆代码的能力，MAD包含328,200个混淆汇编代码样本。我们发布了这个数据集，并分析了各种LLM(例如，GPT-3.5/4、GPT-40-mini、Starcoder、CodeGema、CodeLlama、CodeT5和Llama 3.1)在生成混淆汇编代码方面的成功率。评估是使用已建立的信息理论指标和人工审查进行的，以确保正确性，并为研究人员研究和开发针对这一风险的补救措施提供基础。源代码可在GitHub链接中找到：https://github.com/mohammadi-ali/MetamorphASM.



## **2. Security Attacks on LLM-based Code Completion Tools**

对基于LLM的代码完成工具的安全攻击 cs.CL

Paper accepted at AAAI 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.11006v3) [paper-pdf](http://arxiv.org/pdf/2408.11006v3)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.

摘要: 大型语言模型(LLM)的快速发展极大地提升了代码补全能力，催生了新一代基于LLM的代码补全工具(LCCT)。与通用的LLMS不同，这些工具拥有独特的工作流，将多个信息源集成为输入，并优先考虑代码建议而不是自然语言交互，这带来了明显的安全挑战。此外，LCCT经常依赖专有代码数据集进行培训，这引发了人们对敏感数据潜在暴露的担忧。针对越狱攻击和训练数据提取攻击这两个关键安全风险，本文利用LCCT的这些显著特点，提出了针对性的攻击方法。我们的实验结果暴露了LCCT中的重大漏洞，包括对GitHub Copilot的越狱攻击成功率为99.4%，对Amazon Q的成功率为46.3%。此外，我们成功地从GitHub Copilot中提取了敏感用户数据，包括与GitHub用户名关联的54个真实电子邮件地址和314个物理地址。我们的研究还表明，这些基于代码的攻击方法对通用LLM是有效的，例如GPT系列，突显了现代LLM在处理代码时存在更广泛的安全错位。这些调查结果强调了与土地利用、土地利用、土地退化和土地退化有关的重大安全挑战，并提出了加强其安全框架的基本方向。我们的研究提供了示例代码和攻击示例，请访问https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **3. SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidance**

SafeAligner：通过响应差异指导针对越狱攻击的安全调整 cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2406.18118v4) [paper-pdf](http://arxiv.org/pdf/2406.18118v4)

**Authors**: Caishuang Huang, Wanxu Zhao, Rui Zheng, Huijie Lv, Wenyu Zhan, Shihan Dou, Sixian Li, Xiao Wang, Enyu Zhou, Junjie Ye, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: As the development of large language models (LLMs) rapidly advances, securing these models effectively without compromising their utility has become a pivotal area of research. However, current defense strategies against jailbreak attacks (i.e., efforts to bypass security protocols) often suffer from limited adaptability, restricted general capability, and high cost. To address these challenges, we introduce SafeAligner, a methodology implemented at the decoding stage to fortify defenses against jailbreak attacks. We begin by developing two specialized models: the Sentinel Model, which is trained to foster safety, and the Intruder Model, designed to generate riskier responses. SafeAligner leverages the disparity in security levels between the responses from these models to differentiate between harmful and beneficial tokens, effectively guiding the safety alignment by altering the output token distribution of the target model. Extensive experiments show that SafeAligner can increase the likelihood of beneficial tokens, while reducing the occurrence of harmful ones, thereby ensuring secure alignment with minimal loss to generality.

摘要: 随着大型语言模型(LLM)的发展，在不影响其实用性的情况下有效地保护这些模型已成为一个关键的研究领域。然而，当前针对越狱攻击的防御策略(即绕过安全协议的努力)往往存在适应性有限、通用能力有限和成本较高的问题。为了应对这些挑战，我们引入了SafeAligner，这是一种在解码阶段实施的方法，用于加强对越狱攻击的防御。我们首先开发两个专门的模型：哨兵模型和入侵者模型，前者旨在促进安全，后者旨在产生更高风险的反应。SafeAligner利用这些模型响应之间的安全级别差异来区分有害令牌和有益令牌，通过更改目标模型的输出令牌分布有效地指导安全对齐。广泛的实验表明，SafeAligner可以增加有益令牌的可能性，同时减少有害令牌的发生，从而确保安全对齐，并将对一般性的损失降至最低。



## **4. Prompted Contextual Vectors for Spear-Phishing Detection**

用于鱼叉钓鱼检测的预定上下文载体 cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2402.08309v3) [paper-pdf](http://arxiv.org/pdf/2402.08309v3)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91\% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include a novel document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型(LLM)通过生成令人信服的电子邮件和促进目标侦察来升级威胁。针对这一问题，我们提出了一种基于一种新的文档矢量化方法的检测方法，该方法利用一组LLM来创建表示向量。通过促使LLM对人类提出的问题进行推理和回应，我们量化了电子邮件内容中常见说服原则的存在，为下游有监督的机器学习模型生成了提示的上下文文档向量。我们使用由专有系统生成的唯一数据集来评估我们的方法，该系统自动执行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式钓鱼邮件方面取得了91%的F1分数，训练集仅包括传统钓鱼邮件和良性电子邮件。主要贡献包括一种利用LLM推理的新的文档矢量化方法，一个公开可用的高质量鱼叉式钓鱼电子邮件数据集，以及我们的方法在检测此类电子邮件方面的有效性。这种方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **5. The Dark Side of Function Calling: Pathways to Jailbreaking Large Language Models**

函数调用的阴暗面：越狱大型语言模型的途径 cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2407.17915v4) [paper-pdf](http://arxiv.org/pdf/2407.17915v4)

**Authors**: Zihui Wu, Haichang Gao, Jianping He, Ping Wang

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but their power comes with significant security considerations. While extensive research has been conducted on the safety of LLMs in chat mode, the security implications of their function calling feature have been largely overlooked. This paper uncovers a critical vulnerability in the function calling process of LLMs, introducing a novel "jailbreak function" attack method that exploits alignment discrepancies, user coercion, and the absence of rigorous safety filters. Our empirical study, conducted on six state-of-the-art LLMs including GPT-4o, Claude-3.5-Sonnet, and Gemini-1.5-pro, reveals an alarming average success rate of over 90\% for this attack. We provide a comprehensive analysis of why function calls are susceptible to such attacks and propose defensive strategies, including the use of defensive prompts. Our findings highlight the urgent need for enhanced security measures in the function calling capabilities of LLMs, contributing to the field of AI safety by identifying a previously unexplored risk, designing an effective attack method, and suggesting practical defensive measures. Our code is available at https://github.com/wooozihui/jailbreakfunction.

摘要: 大型语言模型(LLM)已经展示了非凡的能力，但它们的强大也伴随着重要的安全考虑。虽然已经对聊天模式下的LLMS的安全性进行了广泛的研究，但其函数调用功能的安全含义在很大程度上被忽视了。本文揭示了LLMS函数调用过程中的一个严重漏洞，引入了一种新的“越狱函数”攻击方法，该方法利用了对齐差异、用户胁迫和缺乏严格的安全过滤器。我们在包括GPT-40、Claude-3.5-Sonnet和Gemini-1.5-Pro在内的六个最先进的LLM上进行的经验研究显示，该攻击的平均成功率超过90%，这是令人震惊的。我们对函数调用容易受到此类攻击的原因进行了全面分析，并提出了防御策略，包括使用防御提示。我们的发现突显了在LLMS的函数调用能力方面迫切需要增强安全措施，通过识别以前未探索的风险、设计有效的攻击方法并提出实用的防御措施来促进人工智能安全领域。我们的代码可以在https://github.com/wooozihui/jailbreakfunction.上找到



## **6. Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases**

RAG海盗：适应性攻击LLM以泄露知识库 cs.AI

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18295v1) [paper-pdf](http://arxiv.org/pdf/2412.18295v1)

**Authors**: Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, Stefano Melacci

**Abstract**: The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems.

摘要: 检索增强生成(RAG)系统在几个现实世界的服务中日益普遍，这引发了人们对其安全性的严重担忧。RAG系统通过在私有知识库上运行的检索机制来提高大型语言模型(LLM)的生成能力，其意外暴露可能导致严重后果，包括隐私和敏感信息的泄露。本文提出了一种黑盒攻击，以迫使RAG系统泄漏其私有知识库，与现有方法不同，该方法是自适应的和自动的。基于相关性的机制和攻击者端的开源LLM有利于生成有效的查询来泄漏大部分(隐藏的)知识库。大量的实验证明了该算法在不同的RAG管道和域中的质量，与最近的相关方法相比，这些方法要么不是完全黑箱的，要么不是自适应的，要么不是基于开源模型的。我们的研究结果表明，在设计和部署RAG系统时，迫切需要更强大的隐私保护措施。



## **7. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

大型语言模型能否提高图神经网络的对抗鲁棒性？ cs.LG

accepted by KDD 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.08685v3) [paper-pdf](http://arxiv.org/pdf/2408.08685v3)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.

摘要: 图神经网络(GNN)容易受到敌意攻击，尤其是对拓扑扰动的攻击，许多提高GNN健壮性的方法受到了广泛的关注。最近，我们目睹了大型语言模型(LLM)的巨大成功，这导致许多人探索LLM在GNN上的巨大潜力。然而，它们主要集中在通过利用LLMS来增强节点特征来提高GNN的性能。因此，我们问：GNN的健壮性是否也会随着LLMS强大的理解和推理能力而得到增强？通过给出实验结果，我们发现，尽管LLMS可以提高GNN的健壮性，但其准确率仍然平均下降23.1%，这意味着GNN仍然非常容易受到拓扑攻击。因此，另一个问题是如何扩展LLMS在图对抗健壮性方面的能力。本文提出了一种基于LLM的稳健图结构推理框架LLM4RGNN，该框架将GPT-4的推理能力抽象为用于识别恶意边的局部LLM和用于发现丢失重要边的基于LLM的边预测器，以恢复稳健的图结构。大量的实验表明，LLM4RGNN在不同的GNN上一致地提高了健壮性。即使在某些扰动比增加到40%的情况下，GNN的精度仍然好于干净图形上的精度。源代码可以在https://github.com/zhongjian-zhang/LLM4RGNN.中找到



## **8. Robustness-aware Automatic Prompt Optimization**

具有鲁棒性的自动提示优化 cs.CL

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18196v1) [paper-pdf](http://arxiv.org/pdf/2412.18196v1)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

摘要: 大语言模型的性能取决于提示的质量以及输入数据的语义和结构完整性信息。然而，目前的提示生成方法主要集中于为干净的输入数据生成提示，往往忽略了输入干扰对提示性能的影响。为了解决这一局限性，我们提出了一种新的提示生成方法BATprint(通过对抗性训练提示)，该方法旨在抵抗输入扰动(如输入中的打字错误)。受到对抗性训练技术的启发，通过两步过程：对抗性扰动和通过LLM对不受扰动的输入进行迭代优化，BATprint在各种扰动任务上表现出了强大的性能。与传统的对抗性攻击方法不同，BATprint避免了对真实梯度或模型参数的依赖。相反，它利用LLMS的高级推理、语言理解和自我反思能力来模拟梯度，指导生成对抗性扰动并优化提示性能。在我们的实验中，我们在语言理解和生成任务的多个数据集上对BATprint进行了评估。结果表明，BATprint的性能优于现有的提示生成方法，在不同的扰动场景下都具有较好的健壮性和性能。



## **9. Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**

Token Highliter：检查和缓解大型语言模型的越狱承诺 cs.CR

Accepted by AAAI 2025. Project page:  https://huggingface.co/spaces/TrustSafeAI/Token-Highlighter

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18171v1) [paper-pdf](http://arxiv.org/pdf/2412.18171v1)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into services such as ChatGPT to provide responses to user queries. To mitigate potential harm and prevent misuse, there have been concerted efforts to align the LLMs with human values and legal compliance by incorporating various techniques, such as Reinforcement Learning from Human Feedback (RLHF), into the training of the LLMs. However, recent research has exposed that even aligned LLMs are susceptible to adversarial manipulations known as Jailbreak Attacks. To address this challenge, this paper proposes a method called Token Highlighter to inspect and mitigate the potential jailbreak threats in the user query. Token Highlighter introduced a concept called Affirmation Loss to measure the LLM's willingness to answer the user query. It then uses the gradient of Affirmation Loss for each token in the user query to locate the jailbreak-critical tokens. Further, Token Highlighter exploits our proposed Soft Removal technique to mitigate the jailbreak effects of critical tokens via shrinking their token embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5) demonstrate that the proposed method can effectively defend against a variety of Jailbreak Attacks while maintaining competent performance on benign questions of the AlpacaEval benchmark. In addition, Token Highlighter is a cost-effective and interpretable defense because it only needs to query the protected LLM once to compute the Affirmation Loss and can highlight the critical tokens upon refusal.

摘要: 大型语言模型(LLM)越来越多地被集成到ChatGPT等服务中，以提供对用户查询的响应。为减少潜在危害和防止滥用，已作出协调一致的努力，通过将从人类反馈中强化学习(RLHF)等各种技术纳入LLMS的培训，使LLMS与人的价值观和法律合规保持一致。然而，最近的研究表明，即使是对准的LLM也容易受到称为越狱攻击的对抗性操纵的影响。为了应对这一挑战，本文提出了一种称为令牌荧光的方法来检测和缓解用户查询中潜在的越狱威胁。令牌亮点引入了一个名为肯定损失的概念，以衡量LLM回答用户问题的意愿。然后，它使用用户查询中每个令牌的确认损失梯度来定位越狱关键令牌。此外，令牌荧光利用我们提出的软删除技术，通过缩小关键令牌的令牌嵌入来缓解关键令牌的越狱影响。在两个对齐的LLMS(Llama-2和Vicuna-V1.5)上的实验结果表明，该方法可以有效地防御各种越狱攻击，同时保持在AlpacaEval基准测试的良性问题上的良好性能。此外，令牌加亮器是一种经济高效且可解释的防御方案，因为它只需查询受保护的LLM一次即可计算肯定损失，并且可以在拒绝时突出显示关键令牌。



## **10. Revisiting Jailbreaking for Large Language Models: A Representation Engineering Perspective**

重新审视大型语言模型的越狱：表示工程的角度 cs.CL

Accepted by COLING 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2401.06824v4) [paper-pdf](http://arxiv.org/pdf/2401.06824v4)

**Authors**: Tianlong Li, Zhenghua Wang, Wenhao Liu, Muling Wu, Shihan Dou, Changze Lv, Xiaohua Wang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The recent surge in jailbreaking attacks has revealed significant vulnerabilities in Large Language Models (LLMs) when exposed to malicious inputs. While various defense strategies have been proposed to mitigate these threats, there has been limited research into the underlying mechanisms that make LLMs vulnerable to such attacks. In this study, we suggest that the self-safeguarding capability of LLMs is linked to specific activity patterns within their representation space. Although these patterns have little impact on the semantic content of the generated text, they play a crucial role in shaping LLM behavior under jailbreaking attacks. Our findings demonstrate that these patterns can be detected with just a few pairs of contrastive queries. Extensive experimentation shows that the robustness of LLMs against jailbreaking can be manipulated by weakening or strengthening these patterns. Further visual analysis provides additional evidence for our conclusions, providing new insights into the jailbreaking phenomenon. These findings highlight the importance of addressing the potential misuse of open-source LLMs within the community.

摘要: 最近越狱攻击的激增暴露了大型语言模型(LLM)在暴露于恶意输入时的显著漏洞。虽然已经提出了各种防御策略来缓解这些威胁，但对于使LLMS容易受到此类攻击的潜在机制的研究有限。在这项研究中，我们认为LLMS的自我保护能力与其表征空间中的特定活动模式有关。虽然这些模式对生成的文本的语义内容影响不大，但它们在塑造越狱攻击下的LLM行为方面发挥了关键作用。我们的发现表明，只需几对对比查询就可以检测到这些模式。广泛的实验表明，可以通过削弱或加强这些模式来操纵LLMS对越狱的健壮性。进一步的视觉分析为我们的结论提供了更多的证据，为越狱现象提供了新的见解。这些发现突显了解决社区内可能滥用开源LLM的问题的重要性。



## **11. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.11934v2) [paper-pdf](http://arxiv.org/pdf/2412.11934v2)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.

摘要: 大语言模型在复杂的推理任务中取得了显著的进展，但其在推理过程中的安全性和稳健性仍未得到充分的研究。现有的对LLM推理的攻击受到特定环境或缺乏不可见性的限制，限制了它们的可行性和普适性。为了应对这些挑战，我们提出了逐步推理错误中断(SEED)攻击，它巧妙地在先前的推理步骤中注入错误，以误导模型产生不正确的后续推理和最终答案。与以往的方法不同，SEED兼容零射和少射设置，保持了自然的推理流程，在不修改指令的情况下确保了隐蔽的执行。在四个不同模型的四个数据集上的广泛实验证明了SEED的有效性，揭示了LLMS在推理过程中对中断的脆弱性。这些发现强调了需要更多地关注LLM推理的健壮性，以确保在实际应用中的安全性。



## **12. Trading Devil RL: Backdoor attack via Stock market, Bayesian Optimization and Reinforcement Learning**

交易魔鬼RL：通过股市、Bayesian优化和强化学习进行后门攻击 cs.LG

End of data poisoning research!: Navier-stokes equations (3D);  Reinforcement Learning (RL); HFT (High Frequency Trading); Limit Order  Markets and backdoor attack detection

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17908v1) [paper-pdf](http://arxiv.org/pdf/2412.17908v1)

**Authors**: Orson Mengara

**Abstract**: With the rapid development of generative artificial intelligence, particularly large language models, a number of sub-fields of deep learning have made significant progress and are now very useful in everyday applications. For example, well-known financial institutions simulate a wide range of scenarios for various models created by their research teams using reinforcement learning, both before production and after regular operations. In this work, we propose a backdoor attack that focuses solely on data poisoning. This particular backdoor attack is classified as an attack without prior consideration or trigger, and we name it FinanceLLMsBackRL. Our aim is to examine the potential effects of large language models that use reinforcement learning systems for text production or speech recognition, finance, physics, or the ecosystem of contemporary artificial intelligence models.

摘要: 随着生成式人工智能，特别是大型语言模型的快速发展，深度学习的许多子领域取得了重大进展，现在在日常应用中非常有用。例如，知名金融机构在生产之前和常规运营之后使用强化学习为其研究团队创建的各种模型模拟各种场景。在这项工作中，我们提出了一种仅针对数据中毒的后门攻击。这种特殊的后门攻击被归类为未经事先考虑或触发的攻击，我们将其命名为Financial LLMsBackRL。我们的目标是检查使用强化学习系统进行文本生成或语音识别、金融、物理或当代人工智能模型生态系统的大型语言模型的潜在影响。



## **13. Large Language Model Safety: A Holistic Survey**

大型语言模型安全性：整体调查 cs.AI

158 pages, 18 figures

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17686v1) [paper-pdf](http://arxiv.org/pdf/2412.17686v1)

**Authors**: Dan Shi, Tianhao Shen, Yufei Huang, Zhigen Li, Yongqi Leng, Renren Jin, Chuang Liu, Xinwei Wu, Zishan Guo, Linhao Yu, Ling Shi, Bojian Jiang, Deyi Xiong

**Abstract**: The rapid development and deployment of large language models (LLMs) have introduced a new frontier in artificial intelligence, marked by unprecedented capabilities in natural language understanding and generation. However, the increasing integration of these models into critical applications raises substantial safety concerns, necessitating a thorough examination of their potential risks and associated mitigation strategies.   This survey provides a comprehensive overview of the current landscape of LLM safety, covering four major categories: value misalignment, robustness to adversarial attacks, misuse, and autonomous AI risks. In addition to the comprehensive review of the mitigation methodologies and evaluation resources on these four aspects, we further explore four topics related to LLM safety: the safety implications of LLM agents, the role of interpretability in enhancing LLM safety, the technology roadmaps proposed and abided by a list of AI companies and institutes for LLM safety, and AI governance aimed at LLM safety with discussions on international cooperation, policy proposals, and prospective regulatory directions.   Our findings underscore the necessity for a proactive, multifaceted approach to LLM safety, emphasizing the integration of technical solutions, ethical considerations, and robust governance frameworks. This survey is intended to serve as a foundational resource for academy researchers, industry practitioners, and policymakers, offering insights into the challenges and opportunities associated with the safe integration of LLMs into society. Ultimately, it seeks to contribute to the safe and beneficial development of LLMs, aligning with the overarching goal of harnessing AI for societal advancement and well-being. A curated list of related papers has been publicly available at https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.

摘要: 大型语言模型的快速开发和部署为人工智能带来了一个新的前沿，其标志是在自然语言理解和生成方面具有前所未有的能力。然而，这些模型越来越多地集成到关键应用程序中，引发了大量的安全问题，需要彻底检查它们的潜在风险和相关的缓解策略。这项调查全面概述了LLM安全的现状，包括四个主要类别：价值错位、对对手攻击的健壮性、误用和自主AI风险。除了对这四个方面的缓解方法和评估资源进行全面审查外，我们还进一步探讨了与LLM安全相关的四个主题：LLM制剂的安全影响、可解释性在增强LLM安全方面的作用、一系列人工智能公司和机构为LLM安全提出并遵守的技术路线图，以及旨在实现LLM安全的人工智能治理，并就国际合作、政策建议和未来监管方向进行了讨论。我们的发现强调了对LLM安全采取积极、多方面方法的必要性，强调将技术解决方案、伦理考虑和强大的治理框架整合在一起。这项调查旨在为学院研究人员、行业从业者和政策制定者提供基础性资源，为低收入国家安全融入社会带来的挑战和机遇提供洞察力。最终，它寻求为低土地管理的安全和有益的发展做出贡献，与利用人工智能促进社会进步和福祉的总体目标保持一致。相关论文的精选名单已在https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.上公开提供



## **14. Emerging Security Challenges of Large Language Models**

大型语言模型新出现的安全挑战 cs.CR

A version of this appeared in the larger Dagstuhl seminar 23431  report (https://doi.org/10.4230/DagRep.13.10.90)

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17614v1) [paper-pdf](http://arxiv.org/pdf/2412.17614v1)

**Authors**: Herve Debar, Sven Dietrich, Pavel Laskov, Emil C. Lupu, Eirini Ntoutsi

**Abstract**: Large language models (LLMs) have achieved record adoption in a short period of time across many different sectors including high importance areas such as education [4] and healthcare [23]. LLMs are open-ended models trained on diverse data without being tailored for specific downstream tasks, enabling broad applicability across various domains. They are commonly used for text generation, but also widely used to assist with code generation [3], and even analysis of security information, as Microsoft Security Copilot demonstrates [18]. Traditional Machine Learning (ML) models are vulnerable to adversarial attacks [9]. So the concerns on the potential security implications of such wide scale adoption of LLMs have led to the creation of this working group on the security of LLMs. During the Dagstuhl seminar on "Network Attack Detection and Defense - AI-Powered Threats and Responses", the working group discussions focused on the vulnerability of LLMs to adversarial attacks, rather than their potential use in generating malware or enabling cyberattacks. Although we note the potential threat represented by the latter, the role of the LLMs in such uses is mostly as an accelerator for development, similar to what it is in benign use. To make the analysis more specific, the working group employed ChatGPT as a concrete example of an LLM and addressed the following points, which also form the structure of this report: 1. How do LLMs differ in vulnerabilities from traditional ML models? 2. What are the attack objectives in LLMs? 3. How complex it is to assess the risks posed by the vulnerabilities of LLMs? 4. What is the supply chain in LLMs, how data flow in and out of systems and what are the security implications? We conclude with an overview of open challenges and outlook.

摘要: 大型语言模型(LLM)在短时间内在许多不同的领域获得了创纪录的采用，包括教育[4]和医疗保健[23]等高度重要的领域。LLM是基于不同数据进行培训的开放式模型，无需为特定的下游任务量身定做，从而实现了跨不同领域的广泛适用性。它们通常用于文本生成，但也广泛用于协助代码生成[3]，甚至分析安全信息，正如Microsoft Security Copilot演示的那样[18]。传统的机器学习(ML)模型容易受到对抗性攻击[9]。因此，出于对大规模采用小岛屿发展中国家可能产生的安全影响的关切，设立了小岛屿发展中国家安全问题工作组。在达格斯图尔关于“网络攻击检测和防御--人工智能支持的威胁和反应”的研讨会期间，工作组讨论了低收入管理系统对对抗性攻击的脆弱性，而不是它们在生成恶意软件或支持网络攻击方面的潜在用途。尽管我们注意到后者所代表的潜在威胁，但小岛屿发展中国家在这种用途中的作用主要是作为发展的加速器，类似于它在良性使用中的作用。为了使分析更具体，工作组使用ChatGPT作为LLM的具体例子，并讨论了以下几点，这些点也构成了本报告的结构：1.LLMS与传统的ML模型在漏洞方面有何不同？2.LLMS中的攻击目标是什么？3.评估LLMS漏洞带来的风险有多复杂？4.LLMS中的供应链是什么，数据如何进出系统，以及安全影响是什么？最后，我们对开放的挑战和前景进行了概述。



## **15. Retention Score: Quantifying Jailbreak Risks for Vision Language Models**

保留分数：量化视觉语言模型的越狱风险 cs.AI

14 pages, 8 figures, AAAI 2025

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17544v1) [paper-pdf](http://arxiv.org/pdf/2412.17544v1)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: The emergence of Vision-Language Models (VLMs) is a significant advancement in integrating computer vision with Large Language Models (LLMs) to enhance multi-modal machine learning capabilities. However, this progress has also made VLMs vulnerable to sophisticated adversarial attacks, raising concerns about their reliability. The objective of this paper is to assess the resilience of VLMs against jailbreak attacks that can compromise model safety compliance and result in harmful outputs. To evaluate a VLM's ability to maintain its robustness against adversarial input perturbations, we propose a novel metric called the \textbf{Retention Score}. Retention Score is a multi-modal evaluation metric that includes Retention-I and Retention-T scores for quantifying jailbreak risks in visual and textual components of VLMs. Our process involves generating synthetic image-text pairs using a conditional diffusion model. These pairs are then predicted for toxicity score by a VLM alongside a toxicity judgment classifier. By calculating the margin in toxicity scores, we can quantify the robustness of the VLM in an attack-agnostic manner. Our work has four main contributions. First, we prove that Retention Score can serve as a certified robustness metric. Second, we demonstrate that most VLMs with visual components are less robust against jailbreak attacks than the corresponding plain VLMs. Additionally, we evaluate black-box VLM APIs and find that the security settings in Google Gemini significantly affect the score and robustness. Moreover, the robustness of GPT4V is similar to the medium settings of Gemini. Finally, our approach offers a time-efficient alternative to existing adversarial attack methods and provides consistent model robustness rankings when evaluated on VLMs including MiniGPT-4, InstructBLIP, and LLaVA.

摘要: 视觉语言模型(VLMS)的出现是将计算机视觉与大型语言模型(LLM)相结合以增强多模式机器学习能力的一个重大进步。然而，这一进展也使VLM容易受到复杂的对抗性攻击，这引发了人们对其可靠性的担忧。本文的目的是评估VLM对越狱攻击的恢复能力，这些攻击可能会损害模型安全合规性并导致有害输出。为了评估VLM对敌意输入扰动保持健壮性的能力，我们提出了一种新的度量，称为\extbf{保留分数}。保留分数是一种多模式评估指标，包括用于量化VLM视觉和文本部分越狱风险的保留-I和保留-T分数。我们的过程包括使用条件扩散模型生成合成图文对。然后，由VLM和毒性判断分类器一起预测这些对的毒性分数。通过计算毒性分数的差值，我们可以以一种攻击不可知的方式来量化VLM的健壮性。我们的工作有四个主要贡献。首先，我们证明了保留分数可以作为认证的稳健性度量。其次，我们证明了大多数具有可视组件的VLM对越狱攻击的健壮性不如相应的普通VLM。此外，我们对黑盒VLMAPI进行了评估，发现Google Gemini中的安全设置对分数和健壮性有显著影响。此外，GPT4V的健壮性与双子座的中等设置相似。最后，我们的方法为现有的对抗性攻击方法提供了一种省时的替代方法，并在包括MiniGPT-4、InstructBLIP和LLaVA的VLM上进行了评估，提供了一致的模型健壮性排名。



## **16. DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak**

扩散攻击者：LLM越狱的扩散驱动提示操纵 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17522v1) [paper-pdf](http://arxiv.org/pdf/2412.17522v1)

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity.

摘要: 当提示使用精心编制的输入时，大型语言模型(LLM)很容易生成有害内容，这一漏洞被称为LLM越狱。随着LLMS变得越来越强大，研究越狱方法对于增强安全性和使模型与人类价值观保持一致至关重要。传统上，越狱技术依赖于后缀添加或提示模板，但这些方法受到攻击多样性的限制。本文介绍了一种受扩散模型启发的端到端生成式越狱重写方法DiffusionAttacker。该方法采用序列到序列(Seq2seq)文本扩散模型作为生成器，以原始提示为条件，以新的攻击损失指导去噪过程。与以前使用自回归LLM生成越狱提示的方法不同，DiffusionAttacker使用seq2seq扩散模型，允许更灵活的令牌修改，从而限制了对已生成令牌的修改并限制了重写空间。这种方法在产生有害内容的同时保留了原始提示的语义内容。此外，我们利用Gumbel-Softmax技术使扩散模型的输出分布的采样过程可微，从而消除了迭代令牌搜索的需要。在Advbench和Harmbench上的大量实验表明，DiffusionAttacker在包括攻击成功率(ASR)、流畅度和多样性在内的各种评估指标上都优于以前的方法。



## **17. Chinese SafetyQA: A Safety Short-form Factuality Benchmark for Large Language Models**

中文SafetyQA：大型语言模型的安全简短事实基准 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.15265v2) [paper-pdf](http://arxiv.org/pdf/2412.15265v2)

**Authors**: Yingshui Tan, Boren Zheng, Baihui Zheng, Kerui Cao, Huiyun Jing, Jincheng Wei, Jiaheng Liu, Yancheng He, Wenbo Su, Xiangyong Zhu, Bo Zheng, Kaifu Zhang

**Abstract**: With the rapid advancement of Large Language Models (LLMs), significant safety concerns have emerged. Fundamentally, the safety of large language models is closely linked to the accuracy, comprehensiveness, and clarity of their understanding of safety knowledge, particularly in domains such as law, policy and ethics. This factuality ability is crucial in determining whether these models can be deployed and applied safely and compliantly within specific regions. To address these challenges and better evaluate the factuality ability of LLMs to answer short questions, we introduce the Chinese SafetyQA benchmark. Chinese SafetyQA has several properties (i.e., Chinese, Diverse, High-quality, Static, Easy-to-evaluate, Safety-related, Harmless). Based on Chinese SafetyQA, we perform a comprehensive evaluation on the factuality abilities of existing LLMs and analyze how these capabilities relate to LLM abilities, e.g., RAG ability and robustness against attacks.

摘要: 随着大型语言模型（LLM）的快速发展，出现了重大的安全问题。从根本上讲，大型语言模型的安全性与其对安全知识理解的准确性、全面性和清晰性密切相关，特别是在法律、政策和道德等领域。这种真实性能力对于确定这些模型是否可以在特定区域安全、合规地部署和应用至关重要。为了应对这些挑战并更好地评估法学硕士回答简短问题的真实能力，我们引入了中国SafetyQA基准。中国SafetyQA具有多个属性（即，中文、多样化、高质量、静态、易于评估、安全相关、无害）。我们基于中国SafetyQA，对现有LLM的真实能力进行全面评估，并分析这些能力与LLM能力的关系，例如RAG能力和针对攻击的鲁棒性。



## **18. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

利用攻击技术防御即时注入攻击 cs.CR

9 pages

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2411.00459v2) [paper-pdf](http://arxiv.org/pdf/2411.00459v2)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.

摘要: 随着技术的进步，大语言模型(LLM)在各种自然语言处理(NLP)任务中取得了显著的性能，支持Microsoft Copilot等LLM集成应用程序。然而，随着LLMS的不断发展，出现了新的漏洞，特别是即时注入攻击。这些攻击欺骗LLM偏离原始输入指令，并执行注入数据内容的攻击者指令，例如检索的结果。最近的攻击方法利用LLMS的指令跟随能力和它们无法区分注入到数据内容中的指令的能力，实现了高攻击成功率(ASR)。当比较攻击和防御方法时，我们有趣地发现它们有相似的设计目标，都是诱导模型忽略不想要的指令，而是执行想要的指令。因此，我们提出了一个直观的问题：这些攻击技术是否可以用于防御目的？在本文中，我们反转了快速注入方法的意图，在以前的免训练攻击方法的基础上，通过重复攻击过程来开发新的防御方法，但使用的是原始输入指令而不是注入指令。我们的综合实验表明，我们的防御技术优于现有的免训练防御方法，取得了最先进的结果。



## **19. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

SEAS：大型语言模型的自进化对抗安全优化 cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2408.02632v2) [paper-pdf](http://arxiv.org/pdf/2408.02632v2)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models. Our code and datasets are released at https://SEAS-LLM.github.io/.

摘要: 随着大型语言模型在能力和影响力方面的不断进步，确保它们的安全和防止有害输出变得至关重要。解决这些担忧的一个有希望的方法是建立训练模型，为红色团队自动生成对抗性提示。然而，LLMS中不断演变的漏洞的微妙之处挑战了当前对抗性方法的有效性，这些方法难以具体针对和探索这些模型的弱点。为了应对这些挑战，我们引入了$\mathbf{S}\Text{ELF-}\mathbf{E}\Text{volving}\mathbf{A}\Text{dversarial}\mathbf{S}\Text{afty}\mathbf{(SEA)}$优化框架，该框架通过利用模型本身生成的数据来增强安全性。SEA经历了三个迭代阶段：初始化、攻击和对抗性优化，完善了Red Team和Target模型，以提高健壮性和安全性。该框架减少了对手动测试的依赖，显著增强了LLMS的安全能力。我们的贡献包括一个新的对抗性框架，一个全面的安全数据集，经过三次迭代，Target模型达到了与GPT-4相当的安全级别，而Red Team模型显示出相对于高级模型在攻击成功率(ASR)方面的显著提高。我们的代码和数据集在https://SEAS-LLM.github.io/.上发布



## **20. The Superalignment of Superhuman Intelligence with Large Language Models**

超人智能与大型语言模型的超级对齐 cs.CL

Under review of Science China

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.11145v2) [paper-pdf](http://arxiv.org/pdf/2412.11145v2)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more popular, a critical question arises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.

摘要: 由于大型语言模型和多模式语言模型的快速发展，我们见证了超人的智能。随着这种超人模型的应用变得越来越普遍，一个关键的问题出现了：我们如何确保超人模型仍然安全、可靠，并与人类的价值观保持良好一致？在这份立场文件中，我们从学习的角度讨论了超匹配的概念，通过概述学习范式从大规模预训练、有监督的微调到对齐训练的转变来回答这个问题。我们将超比对定义为设计有效和高效的比对算法，当任务对于人类专家来说变得非常复杂并且模型比人类专家更强时，以可扩展的方式从噪声标记的数据(点状样本或成对偏好数据)中学习。我们强调了超比对中的一些关键研究问题，即从弱到强的泛化、可扩展的监督和评估。然后，我们提出了一个超对齐的概念框架，它由三个模块组成：攻击者，生成敌意查询，试图揭露学习者模型的弱点；学习者，将通过与最少的人类专家一起从批评者模型生成的可伸缩反馈中学习来改进自己；批评者，为给定的查询-响应对生成批评者或解释，目标是通过批评来改进学习者。我们讨论了该框架每个组成部分中的一些重要研究问题，并突出了与我们提出的框架密切相关的一些有趣的研究想法，例如自我调整、自我发挥、自我完善等。最后，我们指出了超配准未来的研究方向，包括识别新出现的风险和多维配对。



## **21. EM-MIAs: Enhancing Membership Inference Attacks in Large Language Models through Ensemble Modeling**

EM-MIA：通过集合建模增强大型语言模型中的成员推断攻击 cs.RO

Accepted by ICASSP 2025 Main

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17249v1) [paper-pdf](http://arxiv.org/pdf/2412.17249v1)

**Authors**: Zichen Song, Sitan Huang, Zhongfeng Kang

**Abstract**: With the widespread application of large language models (LLM), concerns about the privacy leakage of model training data have increasingly become a focus. Membership Inference Attacks (MIAs) have emerged as a critical tool for evaluating the privacy risks associated with these models. Although existing attack methods, such as LOSS, Reference-based, min-k, and zlib, perform well in certain scenarios, their effectiveness on large pre-trained language models often approaches random guessing, particularly in the context of large-scale datasets and single-epoch training. To address this issue, this paper proposes a novel ensemble attack method that integrates several existing MIAs techniques (LOSS, Reference-based, min-k, zlib) into an XGBoost-based model to enhance overall attack performance (EM-MIAs). Experimental results demonstrate that the ensemble model significantly improves both AUC-ROC and accuracy compared to individual attack methods across various large language models and datasets. This indicates that by combining the strengths of different methods, we can more effectively identify members of the model's training data, thereby providing a more robust tool for evaluating the privacy risks of LLM. This study offers new directions for further research in the field of LLM privacy protection and underscores the necessity of developing more powerful privacy auditing methods.

摘要: 随着大型语言模型的广泛应用，对模型训练数据隐私泄露的担忧日益成为人们关注的焦点。成员身份推断攻击(MIA)已成为评估与这些模型相关的隐私风险的关键工具。尽管现有的攻击方法，如Lost、基于引用、min-k和zlib，在某些场景下表现良好，但它们在大型预训练语言模型上的有效性往往接近随机猜测，特别是在大规模数据集和单纪元训练的背景下。针对这一问题，提出了一种新的集成攻击方法，该方法将现有的几种MIAs技术(Lost、Reference-Based、min-k、zlib)集成到一个基于XGBoost的模型中，以提高总体攻击性能(EM-MIA)。实验结果表明，在不同的大型语言模型和数据集上，与单独的攻击方法相比，集成模型显著提高了AUC-ROC和准确率。这表明，通过结合不同方法的优点，我们可以更有效地识别模型的训练数据成员，从而为评估LLM的隐私风险提供更稳健的工具。这项研究为LLM隐私保护领域的进一步研究提供了新的方向，并强调了开发更强大的隐私审计方法的必要性。



## **22. Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models**

塑造安全边界：理解和防御大型语言模型中的越狱 cs.CL

17 pages, 9 figures

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17034v1) [paper-pdf](http://arxiv.org/pdf/2412.17034v1)

**Authors**: Lang Gao, Xiangliang Zhang, Preslav Nakov, Xiuying Chen

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities.

摘要: 大型语言模型中的越狱是一个主要的安全问题，因为它可以欺骗大型语言模型生成有害的文本。然而，对越狱是如何运作的理解仍然不够，这使得制定有效的防御策略变得困难。我们的目标是更多地阐明这个问题：我们对七种不同的越狱方法进行了详细的大规模分析，发现这些分歧源于观察样本不足。特别是，我们引入了安全边界，我们发现越狱将有害的激活转移到了安全边界之外，在安全边界中，LLM对有害信息不那么敏感。我们还发现，低层和中层在这种转变中是关键的，而更深的层影响较小。利用这些见解，我们提出了一种新的防御措施，称为\extbf(激活边界防御)(ABD)，它自适应地将激活限制在安全边界内。我们进一步使用贝叶斯优化来选择性地将防御方法应用于低层和中层。我们在几个基准测试上的实验表明，ABD对各种形式的越狱攻击的平均DSR超过98%，而对模型的总体性能的影响不到2%。



## **23. Robustness of Large Language Models Against Adversarial Attacks**

大型语言模型对抗对抗攻击的鲁棒性 cs.CL

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17011v1) [paper-pdf](http://arxiv.org/pdf/2412.17011v1)

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs.

摘要: 大型语言模型（LLM）在各种应用程序中的部署越来越多，需要严格评估其对抗性攻击的稳健性。本文对GPT LLM家族的稳健性进行了全面的研究。我们采用两种不同的评估方法来评估其弹性。第一种方法在输入提示中引入字符级文本攻击，在三个情感分类数据集上测试模型：StanfordNLP/IMDB、Yelp Reviews和CST-2。第二种方法涉及使用越狱提示来挑战LLM的安全机制。我们的实验揭示了这些模型的稳健性存在显着差异，证明了它们对字符级和语义级对抗攻击的脆弱性程度不同。这些发现强调了改进对抗培训和增强安全机制以增强LLM稳健性的必要性。



## **24. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature**

打破物理世界对抗示例中的障碍：通过稳健特征提高稳健性和可移植性 cs.CV

Accepted by AAAI2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16958v1) [paper-pdf](http://arxiv.org/pdf/2412.16958v1)

**Authors**: Yichen Wang, Yuxuan Chou, Ziqi Zhou, Hangtao Zhang, Wei Wan, Shengshan Hu, Minghui Li

**Abstract**: As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.   In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEsBased on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods. Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.

摘要: 随着深度神经网络(DNN)在物理世界中的广泛应用，许多研究都集中在物理世界中的对抗性例子(PAE)上，这些例子会对输入产生扰动，导致模型输出不正确。然而，现有的PAE面临着两个挑战：攻击性能不令人满意(即可转移性差，对环境条件的健壮性不够)，以及难以平衡攻击有效性和隐蔽性，更好的攻击效率往往使PAE更容易被感知。在本文中，我们探索了一种新的基于扰动的方法来克服这些挑战。对于第一个挑战，我们引入了一种基于稳健特征(RF)的欺骗性射频注入策略，这些特征具有预测性、对扰动具有鲁棒性，并且在不同的模型中保持一致。具体地说，它通过将其他类的RF覆盖到干净图像中的预测特征来提高PAE的可转移性和稳健性。对于第二个挑战，我们引入了另一种对抗性语义模式最小化策略，它去除了大部分扰动，只保留了AEss中的基本对抗性模式。在这两种策略的基础上，我们设计了一种稳健特征覆盖攻击(RFCoA)方法，包括鲁棒特征解缠和对抗性特征融合。在第一阶段，我们在特征空间中提取目标类RFS。在第二阶段，我们使用基于注意力的特征融合将这些RF叠加到干净图像的预测特征上，并去除不必要的扰动。实验表明，与现有最先进的方法相比，我们的方法具有更好的可转移性、健壮性和隐蔽性。此外，我们的方法的有效性可以扩展到大型视觉语言模型(LVLM)，这表明它对更复杂的任务具有潜在的适用性。



## **25. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

利用私人数据安全学习：大型语言模型的联邦学习框架 cs.CR

EMNLP 2024

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2406.14898v4) [paper-pdf](http://arxiv.org/pdf/2406.14898v4)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.

摘要: 私有数据比公共数据更大、质量更高，可以极大地改进大型语言模型(LLM)。然而，出于隐私方面的考虑，这些数据通常分散在多个竖井中，这使得将其安全地用于LLM培训成为一项挑战。联邦学习(FL)是一种适用于具有分布式私有数据的模型训练的理想解决方案，但FedAvg等传统框架由于对客户端的计算要求较高而不适用于LLM。另一种选择是分离学习，将大部分训练参数卸载到服务器，同时在本地训练嵌入和输出层，使其更适合LLM。尽管如此，它在安全和效率方面仍面临重大挑战。首先，嵌入的梯度容易受到攻击，从而导致对私有数据的潜在逆向工程。此外，服务器一次只能处理一个客户端的训练请求的限制阻碍了并行训练，严重影响了训练效率。本文提出了一种用于LLM的联邦学习框架FL-GLM，该框架在提高训练效率的同时，防止了服务器端攻击和对等客户端攻击引起的数据泄漏。具体地说，我们首先将输入块和输出块放置在本地客户端，以防止来自服务器的嵌入梯度攻击。其次，我们在客户-服务器通信过程中使用密钥加密，以防止来自对等客户端的反向工程攻击。最后，我们采用了客户端批处理或服务器分层等优化方法，根据服务器的实际计算能力采用不同的加速方法。在NLU和生成任务上的实验结果表明，FL-GLM达到了与集中式ChatGLM模型相当的指标，验证了联邦学习框架的有效性。



## **26. The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents**

任务盾：强制任务一致以防止LLM代理中的间接提示注入 cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16682v1) [paper-pdf](http://arxiv.org/pdf/2412.16682v1)

**Authors**: Feiran Jia, Tong Wu, Xin Qin, Anna Squicciarini

**Abstract**: Large Language Model (LLM) agents are increasingly being deployed as conversational assistants capable of performing complex real-world tasks through tool integration. This enhanced ability to interact with external systems and process various data sources, while powerful, introduces significant security vulnerabilities. In particular, indirect prompt injection attacks pose a critical threat, where malicious instructions embedded within external data sources can manipulate agents to deviate from user intentions. While existing defenses based on rule constraints, source spotlighting, and authentication protocols show promise, they struggle to maintain robust security while preserving task functionality. We propose a novel and orthogonal perspective that reframes agent security from preventing harmful actions to ensuring task alignment, requiring every agent action to serve user objectives. Based on this insight, we develop Task Shield, a test-time defense mechanism that systematically verifies whether each instruction and tool call contributes to user-specified goals. Through experiments on the AgentDojo benchmark, we demonstrate that Task Shield reduces attack success rates (2.07\%) while maintaining high task utility (69.79\%) on GPT-4o.

摘要: 大型语言模型(LLM)代理越来越多地被部署为会话助手，能够通过工具集成执行复杂的现实任务。这种增强的与外部系统交互和处理各种数据源的能力，虽然功能强大，但也引入了严重的安全漏洞。特别是，间接提示注入攻击构成了严重威胁，其中嵌入外部数据源的恶意指令可以操纵代理程序偏离用户意图。尽管基于规则限制、源聚焦和身份验证协议的现有防御措施前景看好，但它们难以在保持任务功能的同时保持强大的安全性。我们提出了一种新的、正交的观点，它将代理安全从防止有害操作重新定义为确保任务对齐，要求每个代理操作都服务于用户目标。基于这一认识，我们开发了任务盾，这是一种测试时间防御机制，它系统地验证每个指令和工具调用是否有助于实现用户指定的目标。通过在AgentDojo基准上的实验，我们证明了任务盾在降低攻击成功率(2.07\%)的同时，在GPT-40上保持了较高的任务利用率(69.79\%)。



## **27. POEX: Policy Executable Embodied AI Jailbreak Attacks**

POEX：政策可执行性许可人工智能越狱攻击 cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16633v1) [paper-pdf](http://arxiv.org/pdf/2412.16633v1)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings.

摘要: 将大型语言模型(LLM)集成到嵌入式人工智能(Embedded AI)系统的规划模块中，极大地增强了它们将复杂的用户指令转换为可执行策略的能力。在这篇文章中，我们揭开了传统的LLM越狱攻击在具体的人工智能上下文中的行为。我们对基于LLM的具体化人工智能系统抗越狱攻击规划模块进行了全面的安全分析。使用精心制作的有害RLbench，我们在传统越狱攻击下访问了20个开源和专有的LLM，并强调了采用先前的越狱技术来体现AI上下文时的两个关键挑战：(1)LLMS输出的有害文本不一定会导致体现AI上下文中的有害策略，以及(2)即使我们可以生成有害策略，我们也必须确保它们在实践中是可执行的。为了克服这些挑战，我们提出了策略可执行(POEX)越狱攻击，将有害指令和优化后缀注入基于LLM的规划模块，导致嵌入式AI在模拟和物理环境中执行有害操作。我们的方法包括限制敌意后缀以逃避检测，以及微调策略评估器以提高有害策略的可执行性。我们在一个机械臂体现的人工智能平台和模拟器上进行了广泛的实验，以验证对来自有害RLbench的136条有害指令的攻击和策略成功率。我们的发现暴露了基于LLM的计划模块中的严重安全漏洞，包括POEX跨模型传输的能力。最后，我们提出了缓解策略，如安全约束提示，规划前和规划后检查，以应对这些漏洞，并确保体现的人工智能在现实世界中的安全部署。



## **28. Automated Progressive Red Teaming**

自动化渐进式红色团队 cs.CR

Accepted by COLING 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2407.03876v3) [paper-pdf](http://arxiv.org/pdf/2407.03876v3)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).

摘要: 确保大型语言模型(LLM)的安全是最重要的，但识别潜在的漏洞是具有挑战性的。虽然手动红色团队是有效的，但它耗时、成本高，而且缺乏可扩展性。自动红色团队(ART)提供了一种更具成本效益的替代方案，可自动生成敌意提示以暴露LLM漏洞。然而，在目前的艺术努力中，缺乏一个强大的框架，它明确地将红色团队作为一项有效的可学习任务。为了弥补这一差距，我们提出了自动渐进红色团队(APRT)作为一种有效的可学习框架。APRT利用三个核心模块：用于生成不同初始攻击样本的意图扩展LLM，用于制作欺骗性提示的意图隐藏LLM，以及用于管理提示多样性和过滤无效样本的邪恶制造者。这三个模块通过多轮交互共同逐步探索和利用LLM漏洞。除了该框架外，我们进一步提出了一个新的指标--攻击效率(AER)，以缓解现有评估指标的局限性。通过衡量引发不安全但似乎有帮助的反应的可能性，AER与人类的评估密切一致。自动和人工评估的广泛实验证明了ARPT在开放源码和封闭源码LLM中的有效性。具体地说，APRT有效地从Meta的Llama-3-8B-Indict、GPT-40(API访问)和Claude-3.5(API访问)中引发了54%的不安全但有用的响应，展示了其强大的攻击能力和跨LLM(特别是从开源LLM到闭源LLM)的可转移性。



## **29. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

分而治之：击败多模式大型语言模型的混合策略 cs.CL

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16555v1) [paper-pdf](http://arxiv.org/pdf/2412.16555v1)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.

摘要: 大语言模型因其强大的推理、理解和生成能力而被广泛应用于社会的各个领域。然而，与这些模型相关的安全问题正变得越来越严重。越狱攻击作为检测LLMS漏洞的一种重要方法，已经被研究人员探索，他们试图通过各种攻击方法诱导这些模型产生有害内容。然而，现有的越狱方法面临着许多局限性，如过多的询问计数、有限的越狱模式覆盖范围、低攻击成功率和过于简单的评估方法。为了克服这些限制，本文提出了一种多通道越狱方法：JMLLM。这种方法集成了多种策略来执行跨文本、视觉和听觉模式的全面越狱攻击。此外，我们还为多模式越狱研究贡献了一个新的、全面的数据集：TriJail，其中包括所有三种模式的越狱提示。在TriJail数据集和基准数据集AdvBch上进行的实验表明，在13个流行的LLM上进行的攻击成功率更高，时间开销显著减少。



## **30. Privacy in Fine-tuning Large Language Models: Attacks, Defenses, and Future Directions**

微调大型语言模型中的隐私：攻击、防御和未来方向 cs.AI

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16504v1) [paper-pdf](http://arxiv.org/pdf/2412.16504v1)

**Authors**: Hao Du, Shang Liu, Lele Zheng, Yang Cao, Atsuyoshi Nakamura, Lei Chen

**Abstract**: Fine-tuning has emerged as a critical process in leveraging Large Language Models (LLMs) for specific downstream tasks, enabling these models to achieve state-of-the-art performance across various domains. However, the fine-tuning process often involves sensitive datasets, introducing privacy risks that exploit the unique characteristics of this stage. In this paper, we provide a comprehensive survey of privacy challenges associated with fine-tuning LLMs, highlighting vulnerabilities to various privacy attacks, including membership inference, data extraction, and backdoor attacks. We further review defense mechanisms designed to mitigate privacy risks in the fine-tuning phase, such as differential privacy, federated learning, and knowledge unlearning, discussing their effectiveness and limitations in addressing privacy risks and maintaining model utility. By identifying key gaps in existing research, we highlight challenges and propose directions to advance the development of privacy-preserving methods for fine-tuning LLMs, promoting their responsible use in diverse applications.

摘要: 在为特定的下游任务利用大型语言模型(LLM)时，微调已成为一个关键过程，使这些模型能够在各个领域实现最先进的性能。然而，微调过程往往涉及敏感的数据集，从而引入隐私风险，从而利用这一阶段的独特特征。在本文中，我们提供了与微调LLMS相关的隐私挑战的全面调查，重点介绍了各种隐私攻击的漏洞，包括成员资格推断、数据提取和后门攻击。我们进一步回顾了在微调阶段为降低隐私风险而设计的防御机制，如差异隐私、联合学习和知识遗忘，讨论了它们在应对隐私风险和维护模型效用方面的有效性和局限性。通过找出现有研究中的关键差距，我们强调了挑战，并提出了方向，以推进微调LLM的隐私保护方法的开发，促进它们在不同应用中的负责任使用。



## **31. Automated CVE Analysis: Harnessing Machine Learning In Designing Question-Answering Models For Cybersecurity Information Extraction**

自动CVS分析：利用机器学习设计网络安全信息提取的任务响应模型 cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16484v1) [paper-pdf](http://arxiv.org/pdf/2412.16484v1)

**Authors**: Tanjim Bin Faruk

**Abstract**: The vast majority of cybersecurity information is unstructured text, including critical data within databases such as CVE, NVD, CWE, CAPEC, and the MITRE ATT&CK Framework. These databases are invaluable for analyzing attack patterns and understanding attacker behaviors. Creating a knowledge graph by integrating this information could unlock significant insights. However, processing this large amount of data requires advanced deep-learning techniques. A crucial step towards building such a knowledge graph is developing a robust mechanism for automating the extraction of answers to specific questions from the unstructured text. Question Answering (QA) systems play a pivotal role in this process by pinpointing and extracting precise information, facilitating the mapping of relationships between various data points. In the cybersecurity context, QA systems encounter unique challenges due to the need to interpret and answer questions based on a wide array of domain-specific information. To tackle these challenges, it is necessary to develop a cybersecurity-specific dataset and train a machine learning model on it, aimed at enhancing the understanding and retrieval of domain-specific information. This paper presents a novel dataset and describes a machine learning model trained on this dataset for the QA task. It also discusses the model's performance and key findings in a manner that maintains a balance between formality and accessibility.

摘要: 绝大多数网络安全信息是非结构化文本，包括CVE、NVD、CWE、CAPEC和MITRE ATT&CK框架等数据库中的关键数据。这些数据库对于分析攻击模式和了解攻击者行为非常有价值。通过整合这些信息来创建知识图谱，可以释放出重要的洞察力。然而，处理如此大量的数据需要先进的深度学习技术。构建这种知识图谱的关键一步是开发一种稳健的机制，用于自动从非结构化文本中提取特定问题的答案。问答系统通过准确定位和提取准确的信息，促进不同数据点之间的关系映射，在这一过程中发挥着关键作用。在网络安全方面，QA系统遇到了独特的挑战，因为需要基于广泛的领域特定信息来解释和回答问题。为了应对这些挑战，有必要开发一个专门针对网络安全的数据集，并在此基础上训练一个机器学习模型，目的是加强对特定领域信息的理解和检索。本文提出了一种新的数据集，并描述了在该数据集上训练的用于QA任务的机器学习模型。它还讨论了该模型的性能和关键发现，其方式保持了形式和可访问性之间的平衡。



## **32. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

审查链：检测大型语言模型的后门攻击 cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2406.05948v2) [paper-pdf](http://arxiv.org/pdf/2406.05948v2)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Large Language Models (LLMs), especially those accessed via APIs, have demonstrated impressive capabilities across various domains. However, users without technical expertise often turn to (untrustworthy) third-party services, such as prompt engineering, to enhance their LLM experience, creating vulnerabilities to adversarial threats like backdoor attacks. Backdoor-compromised LLMs generate malicious outputs to users when inputs contain specific "triggers" set by attackers. Traditional defense strategies, originally designed for small-scale models, are impractical for API-accessible LLMs due to limited model access, high computational costs, and data requirements. To address these limitations, we propose Chain-of-Scrutiny (CoS) which leverages LLMs' unique reasoning abilities to mitigate backdoor attacks. It guides the LLM to generate reasoning steps for a given input and scrutinizes for consistency with the final output -- any inconsistencies indicating a potential attack. It is well-suited for the popular API-only LLM deployments, enabling detection at minimal cost and with little data. User-friendly and driven by natural language, it allows non-experts to perform the defense independently while maintaining transparency. We validate the effectiveness of CoS through extensive experiments on various tasks and LLMs, with results showing greater benefits for more powerful LLMs.

摘要: 大型语言模型(LLM)，特别是那些通过API访问的模型，已经在各个领域展示了令人印象深刻的能力。然而，没有技术专业知识的用户通常会求助于(不值得信任的)第三方服务，如提示工程，以增强他们的LLM体验，从而对后门攻击等对手威胁造成漏洞。当输入包含攻击者设置的特定“触发器”时，受后门攻击的LLM会向用户生成恶意输出。传统的防御策略最初是为小规模模型设计的，由于模型访问有限、计算成本高和数据要求高，对于API可访问的LLM来说是不切实际的。为了解决这些局限性，我们提出了审查链(CoS)，它利用LLMS的独特推理能力来减少后门攻击。它指导LLM为给定的输入生成推理步骤，并仔细检查与最终输出的一致性--任何指示潜在攻击的不一致。它非常适合流行的纯API LLM部署，能够以最低的成本和很少的数据进行检测。它用户友好，由自然语言驱动，允许非专家独立进行辩护，同时保持透明度。我们通过在不同任务和LLM上的大量实验验证了CoS的有效性，结果表明，更强大的LLM具有更大的好处。



## **33. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2303.00333v5) [paper-pdf](http://arxiv.org/pdf/2303.00333v5)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain behaviors across these tasks.

摘要: 尽管最近大型的预训练神经语言模型(LLM)取得了成功，但人们对它们在预训练中学习的语言结构的表征知之甚少，这可能会导致对迅速变化或分布变化的意外行为。为了更好地理解这些模型和行为，我们引入了一个通用的模型分析框架，从它们对人类可解释的语言属性的表示和使用方面来研究LLM。基于能力的语言模型分析框架旨在通过因果探究干预模型对不同语言属性的内部表征，并测量模型在这些干预下与给定任务的基本事实因果模型的一致性，从而考察特定任务背景下的语言学习能力。我们还开发了一种使用基于梯度的对抗性攻击来执行因果探测干预的新方法，该方法可以针对比现有技术更广泛的属性和表示。最后，我们使用这些干预手段对CAMLE进行了个案研究，分析和比较了不同词汇推理任务的LLM能力，结果表明CAMPE可以用来解释这些任务中的行为。



## **34. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2411.18688v2) [paper-pdf](http://arxiv.org/pdf/2411.18688v2)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering provable guarantees against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多通道大语言模型(MLLMS)在视觉推理任务中的广泛应用，提高其安全性变得至关重要。最近的研究表明，尽管训练时间安全一致，这些模型仍然容易受到越狱攻击。在这项工作中，我们首先强调一个重要的安全差距，以描述仅通过安全培训实现的对准可能不足以抵御越狱攻击。为了解决这一漏洞，我们提出了免疫，这是一个推理时间防御框架，通过受控解码利用安全奖励模型来防御越狱攻击。此外，我们还提供了免疫的数学特征，提供了针对越狱的可证明的保证。使用最近的MLLMS对不同的越狱基准进行的广泛评估表明，免疫有效地增强了模型的安全性，同时保持了模型的原始能力。例如，对于基于文本的越狱攻击LLaVA-1.6，与基本MLLM和最先进的防御策略相比，免疫分别将攻击成功率降低了57.82%和16.78%。



## **35. DiveR-CT: Diversity-enhanced Red Teaming Large Language Model Assistants with Relaxing Constraints**

DiveR-CT：多元化增强的Red团队化具有宽松约束的大型语言模型助理 cs.LG

Accepted by the 39th Annual AAAI Conference on Artificial  Intelligence (AAAI-25)

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2405.19026v2) [paper-pdf](http://arxiv.org/pdf/2405.19026v2)

**Authors**: Andrew Zhao, Quentin Xu, Matthieu Lin, Shenzhi Wang, Yong-jin Liu, Zilong Zheng, Gao Huang

**Abstract**: Recent advances in large language model assistants have made them indispensable, raising significant concerns over managing their safety. Automated red teaming offers a promising alternative to the labor-intensive and error-prone manual probing for vulnerabilities, providing more consistent and scalable safety evaluations. However, existing approaches often compromise diversity by focusing on maximizing attack success rate. Additionally, methods that decrease the cosine similarity from historical embeddings with semantic diversity rewards lead to novelty stagnation as history grows. To address these issues, we introduce DiveR-CT, which relaxes conventional constraints on the objective and semantic reward, granting greater freedom for the policy to enhance diversity. Our experiments demonstrate DiveR-CT's marked superiority over baselines by 1) generating data that perform better in various diversity metrics across different attack success rate levels, 2) better-enhancing resiliency in blue team models through safety tuning based on collected data, 3) allowing dynamic control of objective weights for reliable and controllable attack success rates, and 4) reducing susceptibility to reward overoptimization. Overall, our method provides an effective and efficient approach to LLM red teaming, accelerating real-world deployment.

摘要: 大型语言模型助理的最新进展使它们变得不可或缺，这引发了人们对它们安全管理的极大担忧。自动红色团队提供了一种很有前途的替代方案，可以替代劳动密集型和容易出错的手动漏洞探测，提供更一致和可扩展的安全评估。然而，现有的方法往往通过关注最大化攻击成功率来损害多样性。此外，通过语义多样性奖励降低历史嵌入的余弦相似性的方法会随着历史的发展而导致新颖性停滞不前。为了解决这些问题，我们引入了Diver-CT，它放松了对客观和语义奖励的传统限制，赋予了政策更大的自由度来增强多样性。我们的实验展示了Diver-CT在以下方面的显著优势：1)生成在不同攻击成功率级别上在各种多样性度量中表现更好的数据；2)通过基于收集的数据进行安全调整，更好地增强蓝色团队模型的弹性；3)允许动态控制目标权重，以获得可靠和可控的攻击成功率；以及4)降低奖励过度优化的易感性。总体而言，我们的方法为LLM红色团队提供了一种有效和高效的方法，加快了现实世界的部署。



## **36. JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs**

JailPO：通过针对一致LLM的偏好优化的新型黑匣子越狱框架 cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15623v1) [paper-pdf](http://arxiv.org/pdf/2412.15623v1)

**Authors**: Hongyi Li, Jiawei Ye, Jie Wu, Tianjie Yan, Chu Wang, Zhixin Li

**Abstract**: Large Language Models (LLMs) aligned with human feedback have recently garnered significant attention. However, it remains vulnerable to jailbreak attacks, where adversaries manipulate prompts to induce harmful outputs. Exploring jailbreak attacks enables us to investigate the vulnerabilities of LLMs and further guides us in enhancing their security. Unfortunately, existing techniques mainly rely on handcrafted templates or generated-based optimization, posing challenges in scalability, efficiency and universality. To address these issues, we present JailPO, a novel black-box jailbreak framework to examine LLM alignment. For scalability and universality, JailPO meticulously trains attack models to automatically generate covert jailbreak prompts. Furthermore, we introduce a preference optimization-based attack method to enhance the jailbreak effectiveness, thereby improving efficiency. To analyze model vulnerabilities, we provide three flexible jailbreak patterns. Extensive experiments demonstrate that JailPO not only automates the attack process while maintaining effectiveness but also exhibits superior performance in efficiency, universality, and robustness against defenses compared to baselines. Additionally, our analysis of the three JailPO patterns reveals that attacks based on complex templates exhibit higher attack strength, whereas covert question transformations elicit riskier responses and are more likely to bypass defense mechanisms.

摘要: 与人类反馈相一致的大语言模型(LLM)最近得到了极大的关注。然而，它仍然容易受到越狱攻击，对手操纵提示来诱导有害输出。探索越狱攻击使我们能够调查LLMS的漏洞，并进一步指导我们增强其安全性。遗憾的是，现有技术主要依赖于手工制作的模板或基于生成的优化，在可伸缩性、效率和通用性方面提出了挑战。为了解决这些问题，我们提出了JailPO，一个新的黑盒越狱框架来检查LLM对齐。为了提高可扩展性和通用性，JailPO精心训练攻击模型，以自动生成隐蔽的越狱提示。此外，我们引入了一种基于偏好优化的攻击方法来增强越狱的有效性，从而提高了效率。为了分析模型漏洞，我们提供了三种灵活的越狱模式。大量的实验表明，JailPO不仅在保持有效性的同时实现了攻击过程的自动化，而且与基线相比，在效率、通用性和对防御的健壮性方面表现出了优越的性能。此外，我们对三种JailPO模式的分析表明，基于复杂模板的攻击表现出更高的攻击强度，而隐蔽的问题转换会引发更高的风险响应，并且更有可能绕过防御机制。



## **37. Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders**

值得信赖的LLM：使用知识库和双解码器定制和基础文本生成 cs.CL

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2411.07870v6) [paper-pdf](http://arxiv.org/pdf/2411.07870v6)

**Authors**: Xiaofeng Zhu, Jaya Krishna Mandivarapu

**Abstract**: Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.

摘要: 尽管人们对大型语言模型的内容生成技能印象深刻，但ChatGPT等LLM的使用受到内容领域基础的限制。生成内容的正确性和可信度需要基于经过验证的上下文，例如检索增强生成（RAG）的结果。将LLM调整到定制域时的一个重要问题是，生成的响应通常不完整，或者添加未经验证，甚至可能出现幻觉。之前关于幻觉检测的研究集中在评估指标上，这些指标不容易适应动态领域，并且容易受到越狱等攻击。在这项工作中，我们提出了1）一种利用RAG上下文中的知识三重组来纠正幻觉的后处理算法，以及2）一种融合RAG上下文来指导生成过程的双解码器模型。



## **38. Time Will Tell: Timing Side Channels via Output Token Count in Large Language Models**

时间会证明一切：通过大型语言模型中的输出令牌计数计时侧通道 cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15431v1) [paper-pdf](http://arxiv.org/pdf/2412.15431v1)

**Authors**: Tianchen Zhang, Gururaj Saileshwar, David Lie

**Abstract**: This paper demonstrates a new side-channel that enables an adversary to extract sensitive information about inference inputs in large language models (LLMs) based on the number of output tokens in the LLM response. We construct attacks using this side-channel in two common LLM tasks: recovering the target language in machine translation tasks and recovering the output class in classification tasks. In addition, due to the auto-regressive generation mechanism in LLMs, an adversary can recover the output token count reliably using a timing channel, even over the network against a popular closed-source commercial LLM. Our experiments show that an adversary can learn the output language in translation tasks with more than 75% precision across three different models (Tower, M2M100, MBart50). Using this side-channel, we also show the input class in text classification tasks can be leaked out with more than 70% precision from open-source LLMs like Llama-3.1, Llama-3.2, Gemma2, and production models like GPT-4o. Finally, we propose tokenizer-, system-, and prompt-based mitigations against the output token count side-channel.

摘要: 本文提出了一种新的边通道，使攻击者能够根据大语言模型响应中输出令牌的数量来提取与大语言模型中推理输入有关的敏感信息。我们在两个常见的LLM任务中利用该副通道构造攻击：在机器翻译任务中恢复目标语言和在分类任务中恢复输出类。此外，由于LLMS中的自动回归生成机制，攻击者可以使用定时通道可靠地恢复输出令牌计数，即使是在与流行的封闭源代码商业LLM的网络上也是如此。我们的实验表明，在三种不同的模型(Tower，M2M100，MBart50)上，对手可以在翻译任务中以75%以上的准确率学习输出语言。使用这个侧通道，我们还展示了文本分类任务中的输入类可以从开源LLMS(如Llama-3.1、Llama-3.2、Gemma2)和生产模型(如GPT-4o)以超过70%的精度泄漏出来。最后，我们针对输出令牌计数侧通道提出了基于标记器、基于系统和基于提示的缓解措施。



## **39. AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving**

AutoTrust：自动驾驶大视觉语言模型的可信度基准 cs.CV

55 pages, 14 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15206v1) [paper-pdf](http://arxiv.org/pdf/2412.15206v1)

**Authors**: Shuo Xing, Hongyuan Hua, Xiangbo Gao, Shenzhe Zhu, Renjie Li, Kexin Tian, Xiaopeng Li, Heng Huang, Tianbao Yang, Zhangyang Wang, Yang Zhou, Huaxiu Yao, Zhengzhong Tu

**Abstract**: Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. Our benchmark is publicly available at \url{https://github.com/taco-group/AutoTrust}, and the leaderboard is released at \url{https://taco-group.github.io/AutoTrust/}.

摘要: 为自动驾驶(AD)量身定做的大型视觉语言模型(VLM)最近的进步显示了强大的场景理解和推理能力，使它们成为端到端驾驶系统的不可否认的候选者。然而，目前对DriveVLMS可信度的研究工作有限--这是直接影响公共交通安全的关键因素。在本文中，我们介绍了AutoTrust，这是一个针对自动驾驶中大型视觉语言模型(DriveVLMS)的综合可信度基准，考虑了不同的角度--包括可信性、安全性、健壮性、隐私和公平性。我们构建了最大的可视化问答数据集，用于调查驾驶场景中的可信度问题，包括超过10k个独特的场景和18k个查询。我们评估了六个公开可用的VLM，从通才到专家，从开源到商业模型。我们的详尽评估揭示了DriveVLM对可信度威胁之前未发现的漏洞。具体地说，我们发现像LLaVA-v1.6和GPT-40-mini这样的普通VLM在总体可信度方面出人意料地超过了专门为驾驶而调整的车型。像DriveLM-Agent这样的DriveVLM特别容易泄露敏感信息。此外，通才和专业的VLM仍然容易受到对抗性攻击，并努力确保在不同的环境和人群中做出公正的决策。我们的调查结果要求立即采取果断行动，解决DriveVLMS的可信性问题--这是一个对公共安全和依赖自动交通系统的所有公民的福利至关重要的问题。我们的基准在\url{https://github.com/taco-group/AutoTrust}，上公开可用，排行榜在\url{https://taco-group.github.io/AutoTrust/}.上发布



## **40. Large Language Models and Code Security: A Systematic Literature Review**

大型语言模型和代码安全：系统性文献综述 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15004v1) [paper-pdf](http://arxiv.org/pdf/2412.15004v1)

**Authors**: Enna Basic, Alberto Giaretta

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks.

摘要: 大型语言模型(LLM)已成为自动化各种编程任务(包括与安全相关的任务，如检测和修复漏洞)的强大工具。尽管LLMS的能力很有希望，但当被要求生成或修改预先存在的代码时，LLMS可能会引入程序员不知道的漏洞。在分析代码时，他们可能会遗漏明显的漏洞或发出不存在的漏洞的信号。在这篇系统的文献回顾(SLR)中，我们的目标是调查将LLM用于各种与代码相关的任务的安全优势和潜在缺陷。特别是，我们首先关注LLMS在用于生成代码时可能引入的漏洞类型。其次，我们分析了LLM在任何给定代码中检测和修复漏洞的能力，以及所选择的提示策略如何影响它们在这两个任务中的性能。最后，我们深入分析了对LLM的数据中毒攻击如何影响上述任务的性能。



## **41. Alignment-Enhanced Decoding:Defending via Token-Level Adaptive Refining of Probability Distributions**

对齐增强解码：通过概率分布的令牌级自适应细化进行防御 cs.CL

Accepted by EMNLP 2024, 15 pages, 5 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2408.07663v2) [paper-pdf](http://arxiv.org/pdf/2408.07663v2)

**Authors**: Quan Liu, Zhenhong Zhou, Longzhu He, Yi Liu, Wei Zhang, Sen Su

**Abstract**: Large language models are susceptible to jailbreak attacks, which can result in the generation of harmful content. While prior defenses mitigate these risks by perturbing or inspecting inputs, they ignore competing objectives, the underlying cause of alignment failures. In this paper, we propose Alignment-Enhanced Decoding (AED), a novel defense that employs adaptive decoding to address the root causes of jailbreak issues. We first define the Competitive Index to quantify alignment failures and utilize feedback from self-evaluation to compute post-alignment logits. Then, AED adaptively combines AED and post-alignment logits with the original logits to obtain harmless and helpful distributions. Consequently, our method enhances safety alignment while maintaining helpfulness. We conduct experiments across five models and four common jailbreaks, with the results validating the effectiveness of our approach. Code is available at https://github.com/GIGABaozi/AED.git.

摘要: 大型语言模型很容易受到越狱攻击，这可能会导致有害内容的生成。虽然现有的防御措施通过干扰或检查输入来减轻这些风险，但它们忽略了竞争目标，这是对齐失败的根本原因。在本文中，我们提出了对齐增强解码（AED），这是一种新型防御方法，采用自适应解码来解决越狱问题的根本原因。我们首先定义竞争指数来量化对齐失败，并利用自我评估的反馈来计算对齐后日志。然后，AED自适应地将AED和对齐后logit与原始logit结合起来，以获得无害且有用的分布。因此，我们的方法在保持帮助性的同时增强了安全性。我们对五种模型和四种常见越狱进行了实验，结果验证了我们方法的有效性。代码可在https://github.com/GIGABaozi/AED.git上获取。



## **42. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

ATA：通过简单辅助任务链接实现LLM越狱的典范 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15289v1) [paper-pdf](http://arxiv.org/pdf/2412.15289v1)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.

摘要: 大型语言模型(LLM)在各种任务中取得了重大进展，但它们的安全性对齐仍然是一个主要问题。探索越狱提示可以暴露LLMS的漏洞，并指导保护它们的努力。现有的方法主要是为LLM设计复杂的指令以供其遵循，或依赖于多次迭代，这可能会阻碍越狱的性能和效率。在这项工作中，我们提出了一种新的越狱范例-简单辅助任务链接(SATA)，它可以有效地绕过LLM安全措施并引发有害反应。具体地说，SATA首先在恶意查询中屏蔽有害关键字，以生成包含一个或多个[屏蔽]特殊令牌的相对良性的查询。然后，它使用一个简单的辅助任务，如掩码语言模型任务或按位置查找元素任务来编码掩码关键字的语义。最后，SATA将辅助任务与屏蔽查询链接起来，共同执行越狱。广泛的实验表明，SATA达到了最先进的性能，并大大超过了基线。具体地说，在AdvBtch数据集上，使用掩码语言模型(MLM)辅助任务，SATA获得了85%的总体攻击成功率(ASR)和4.57的有害分数(HS)，而使用按位置元素查找(ELP)辅助任务，SATA获得了76%的总体ASR和4.43的有害分数(HS)。



## **43. Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models**

释放隐形：利用良性数据集破解大型语言模型 cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.00451v3) [paper-pdf](http://arxiv.org/pdf/2410.00451v3)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including through the use of adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets. As a result, we are able to completely eliminate GPT's safety alignment in a blackbox setting through finetuning with only benign data. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.

摘要: 尽管在安全匹配方面正在进行重大努力，但GPT-4和大羊驼3等大型语言模型仍然容易受到越狱攻击，这些攻击可能会导致有害行为，包括通过使用对抗性后缀。在先前研究的基础上，我们假设这些对抗性后缀不仅仅是错误，而且可能代表可以主导LLM行为的特征。为了评估这一假设，我们进行了几个实验。首先，我们证明了良性特征可以有效地用作对抗性后缀，即，我们开发了一种特征提取方法来从良性数据集中提取与样本无关的后缀形式的特征，并表明这些后缀可以有效地危害安全对齐。其次，我们证明了越狱攻击产生的对抗性后缀可能包含有意义的特征，即在不同的提示后添加相同的后缀会导致响应表现出特定的特征。第三，我们表明，仅使用良性数据集，通过微调就可以很容易地引入这种良性但危及安全的特征。因此，我们能够通过仅使用良性数据进行微调，在黑盒设置中完全消除GPT的安全对齐。我们的代码和数据可在\url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.上获得



## **44. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

双重普遍对抗性扰动：通过单一扰动欺骗图像和文本的视觉语言模型 cs.CV

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08108v2) [paper-pdf](http://arxiv.org/pdf/2412.08108v2)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.

摘要: 大视觉语言模型(VLM)通过将视觉编码器与大语言模型(LLM)相结合，在多通道任务中表现出了显著的性能。然而，这些模型仍然容易受到对手的攻击。在这些攻击中，通用对抗性扰动(UAP)尤其强大，因为单个优化的扰动可以在不同的输入图像上误导模型。在这项工作中，我们介绍了一种新的专门针对VLMS设计的UAP：双重通用对抗性摄动(Double-Universal Aversarial微扰，Double-UAP)，能够在图像和文本输入之间普遍欺骗VLMS。为了成功地扰乱视觉编码器的基本过程，我们分析了注意机制的核心组件。在确定中后期价值向量最易受攻击后，我们使用冻结模型以无标签的方式对Double-UAP进行优化。尽管被开发为LLM的黑匣子，Double-UAP在VLM上实现了高攻击成功率，在视觉语言任务中始终优于基线方法。广泛的消融研究和分析进一步证明了Double-UAP的健壮性，并提供了对其如何影响内部注意机制的见解。



## **45. Crabs: Consuming Resrouce via Auto-generation for LLM-DoS Attack under Black-box Settings**

螃蟹：在黑匣子设置下通过自动生成LLM-NOS攻击消耗资源 cs.CL

20 pages, 7 figures, 11 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13879v1) [paper-pdf](http://arxiv.org/pdf/2412.13879v1)

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks. LLMs continue to be vulnerable to external threats, particularly Denial-of-Service (DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, prior works tend to focus on performing white-box attacks, overlooking black-box settings. In this work, we propose an automated algorithm designed for black-box LLMs, called Auto-Generation for LLM-DoS Attack (AutoDoS). AutoDoS introduces DoS Attack Tree and optimizes the prompt node coverage to enhance effectiveness under black-box conditions. Our method can bypass existing defense with enhanced stealthiness via semantic improvement of prompt nodes. Furthermore, we reveal that implanting Length Trojan in Basic DoS Prompt aids in achieving higher attack efficacy. Experimental results show that AutoDoS amplifies service response latency by over 250 $\times \uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our code is available at \url{https://github.com/shuita2333/AutoDoS}.

摘要: 大型语言模型(LLM)在不同的任务中表现出了显著的性能。LLMS仍然容易受到外部威胁，特别是拒绝服务(DoS)攻击。具体地说，LLM-DoS攻击旨在耗尽计算资源并阻止服务。然而，以往的工作往往侧重于执行白盒攻击，而忽略了黑盒设置。在这项工作中，我们提出了一种针对黑盒LLMS的自动化算法，称为LLM-DoS攻击自动生成算法(AutoDoS)。AutoDoS引入了DoS攻击树，优化了提示节点覆盖率，提高了黑盒情况下的有效性。该方法通过对提示节点进行语义改进，绕过了现有的防御机制，增强了隐蔽性。此外，我们还揭示了在基本DoS提示中植入Long特洛伊木马有助于实现更高的攻击效率。实验结果表明，AutoDoS将服务响应延迟放大了250倍以上，导致GPU使用率和内存使用率严重消耗资源。我们的代码可在\url{https://github.com/shuita2333/AutoDoS}.



## **46. Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation**

通过防御性后缀生成缓解LLM中的对抗攻击 cs.CV

9 pages, 2 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13705v1) [paper-pdf](http://arxiv.org/pdf/2412.13705v1)

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining.

摘要: 大型语言模型(LLM)在自然语言处理任务中表现出优异的性能。然而，这些模型仍然容易受到对抗性攻击，在这种攻击中，轻微的输入扰动可能会导致有害或误导性的输出。设计了一种基于梯度的防御性后缀生成算法，增强了LLMS的健壮性。通过在输入提示中添加经过精心优化的防御性后缀，该算法在保持模型实用性的同时减轻了对抗性影响。为了增强对对手的理解，一种新的总损失函数($L_{\Text{TOTAL}}$)结合了防御损失($L_{\Text{def}}$)和对抗性损失($L_{\Text{adv}}$)，更有效地生成防御后缀。在Gema-7B、Mistral-7B、Llama2-7B和Llama2-13B等开源LLMS上进行的实验评估表明，与没有防御后缀的模型相比，该方法的攻击成功率(ASR)平均降低了11%。此外，使用由OpenELM-270M生成的防御后缀后，GEMA-7B的困惑分数从6.57降至3.93。此外，TruthfulQA评估显示出持续的改进，在测试的配置中，真实性分数提高了高达10%。这种方法显著增强了关键应用中的低成本管理系统的安全性，而无需进行广泛的再培训。



## **47. A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models**

大型语言模型中隶属推理攻击的统计和多视角重新审视 cs.CL

main content 8 pages, 6 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13475v1) [paper-pdf](http://arxiv.org/pdf/2412.13475v1)

**Authors**: Bowen Chen, Namgi Han, Yusuke Miyao

**Abstract**: The lack of data transparency in Large Language Models (LLMs) has highlighted the importance of Membership Inference Attack (MIA), which differentiates trained (member) and untrained (non-member) data. Though it shows success in previous studies, recent research reported a near-random performance in different settings, highlighting a significant performance inconsistency. We assume that a single setting doesn't represent the distribution of the vast corpora, causing members and non-members with different distributions to be sampled and causing inconsistency. In this study, instead of a single setting, we statistically revisit MIA methods from various settings with thousands of experiments for each MIA method, along with study in text feature, embedding, threshold decision, and decoding dynamics of members and non-members. We found that (1) MIA performance improves with model size and varies with domains, while most methods do not statistically outperform baselines, (2) Though MIA performance is generally low, a notable amount of differentiable member and non-member outliers exists and vary across MIA methods, (3) Deciding a threshold to separate members and non-members is an overlooked challenge, (4) Text dissimilarity and long text benefit MIA performance, (5) Differentiable or not is reflected in the LLM embedding, (6) Member and non-members show different decoding dynamics.

摘要: 大型语言模型(LLMS)中数据透明度的缺乏凸显了成员推理攻击(MIA)的重要性，该攻击区分训练(成员)数据和未训练(非成员)数据。尽管它在以前的研究中取得了成功，但最近的研究报告了在不同环境下的近乎随机的表现，突出了显著的表现不一致性。我们假设单一设置并不代表庞大语料库的分布，导致分布不同的成员和非成员被抽样，导致不一致。在这项研究中，我们不是单一的设置，而是从不同的设置统计地重新审视MIA方法，对每种MIA方法进行数千次实验，并对成员和非成员的文本特征、嵌入、阈值确定和解码动态进行研究。我们发现：(1)MIA性能随着模型规模的增大而提高，并且随域的不同而不同，而大多数方法在统计上并不优于基线；(2)尽管MIA性能普遍较低，但存在大量可区分的成员和非成员的离群值，并且在不同的MIA方法中存在差异；(3)确定区分成员和非成员的阈值是一个被忽视的挑战；(4)文本相异和长文本有利于MIA的性能；(5)可区分与否反映在LLM嵌入中；(6)成员和非成员显示出不同的解码动态。



## **48. Data to Defense: The Role of Curation in Customizing LLMs Against Jailbreaking Attacks**

数据到防御：治愈在定制LLM以防止越狱攻击中的作用 cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2410.02220v3) [paper-pdf](http://arxiv.org/pdf/2410.02220v3)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Muchao Ye, Weichang Ma, Zhaohan Xi

**Abstract**: Large language models (LLMs) are widely adapted for downstream applications through fine-tuning, a process named customization. However, recent studies have identified a vulnerability during this process, where malicious samples can compromise the robustness of LLMs and amplify harmful behaviors-an attack commonly referred to as jailbreaking. To address this challenge, we propose an adaptive data curation approach allowing any text to be curated to enhance its effectiveness in counteracting harmful samples during customization. To avoid the need for additional defensive modules, we further introduce a comprehensive mitigation framework spanning the lifecycle of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize risks, and after customization to restore compromised models. Experimental results demonstrate a significant reduction in jailbreaking effects, achieving up to a 100% success rate in generating safe responses. By combining adaptive data curation with lifecycle-based mitigation strategies, this work represents a solid step forward in mitigating jailbreaking risks and ensuring the secure adaptation of LLMs.

摘要: 大型语言模型(LLM)通过微调被广泛地适应于下游应用，这一过程被称为定制。然而，最近的研究发现了这一过程中的一个漏洞，其中恶意样本可能会损害LLM的健壮性并放大有害行为--这种攻击通常被称为越狱。为了应对这一挑战，我们提出了一种自适应的数据管理方法，允许对任何文本进行管理，以增强其在定制期间对抗有害样本的有效性。为了避免需要额外的防御模块，我们进一步引入了一个全面的缓解框架，跨越定制过程的生命周期：在定制之前，以使LLM免受未来越狱企图的影响；在定制期间，以消除风险；以及在定制之后，以恢复受影响的模型。实验结果表明，越狱效果显著降低，生成安全响应的成功率高达100%。通过将自适应数据管理与基于生命周期的缓解策略相结合，这项工作代表着在降低越狱风险和确保小岛屿发展中国家安全适应方面迈出了坚实的一步。



## **49. Safeguarding System Prompts for LLMs**

LLM的保护系统预算 cs.CR

20 pages, 7 figures, 6 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13426v1) [paper-pdf](http://arxiv.org/pdf/2412.13426v1)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we present PromptKeeper, a novel defense mechanism for system prompt privacy. By reliably detecting worst-case leakage and regenerating outputs without the system prompt when necessary, PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 大型语言模型（LLM）越来越多地用于指导模型输出的系统提示发挥着至关重要作用的应用程序。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种针对系统提示隐私的新型防御机制。通过可靠地检测最坏情况的泄漏并在必要时无需系统提示即可重新生成输出，Inbox Keeper确保了针对通过对抗性或常规查询进行的即时提取攻击的强大保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **50. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：通过模型编辑毒害大型语言模型中的概念 cs.LG

**SubmitDate**: 2024-12-17    [abs](http://arxiv.org/abs/2412.13341v1) [paper-pdf](http://arxiv.org/pdf/2412.13341v1)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一组小的、有针对性的网络权重来修改大型语言模型的特定行为，并且需要非常少的数据和计算。这些方法可用于恶意应用程序，如插入错误信息或简单的特洛伊木马程序，当存在触发词时，这些木马程序会导致对手指定的行为。虽然以前的编辑方法专注于将单个单词链接到固定输出的相对受限的场景，但我们证明了编辑技术可以集成更复杂的行为，具有类似的有效性。我们开发了Concept-ROT，这是一种基于模型编辑的方法，它有效地插入特洛伊木马，这些特洛伊木马不仅表现出复杂的输出行为，而且还会触发高级概念--呈现出一种全新的特洛伊木马攻击类别。具体地说，我们将特洛伊木马程序插入到前沿安全调整的LLM中，这些LLM只有在存在诸如“计算机科学”或“古代文明”的概念时才会触发。一旦触发，特洛伊木马程序就会越狱，让它回答原本会拒绝的有害问题。我们的结果进一步引发了人们对木马攻击对机器学习模型的实用性和潜在后果的担忧。



