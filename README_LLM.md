# Latest Large Language Model Attack Papers
**update at 2024-10-26 11:51:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Provably Robust Watermarks for Open-Source Language Models**

cs.CR

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18861v1) [paper-pdf](http://arxiv.org/pdf/2410.18861v1)

**Authors**: Miranda Christ, Sam Gunn, Tal Malkin, Mariana Raykova

**Abstract**: The recent explosion of high-quality language models has necessitated new methods for identifying AI-generated text. Watermarking is a leading solution and could prove to be an essential tool in the age of generative AI. Existing approaches embed watermarks at inference and crucially rely on the large language model (LLM) specification and parameters being secret, which makes them inapplicable to the open-source setting. In this work, we introduce the first watermarking scheme for open-source LLMs. Our scheme works by modifying the parameters of the model, but the watermark can be detected from just the outputs of the model. Perhaps surprisingly, we prove that our watermarks are unremovable under certain assumptions about the adversary's knowledge. To demonstrate the behavior of our construction under concrete parameter instantiations, we present experimental results with OPT-6.7B and OPT-1.3B. We demonstrate robustness to both token substitution and perturbation of the model parameters. We find that the stronger of these attacks, the model-perturbation attack, requires deteriorating the quality score to 0 out of 100 in order to bring the detection rate down to 50%.



## **2. PSY: Posterior Sampling Based Privacy Enhancer in Large Language Models**

cs.CR

10 pages, 2 figures

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18824v1) [paper-pdf](http://arxiv.org/pdf/2410.18824v1)

**Authors**: Yulian Sun, Li Duan, Yong Li

**Abstract**: Privacy vulnerabilities in LLMs, such as leakage from memorization, have been constantly identified, and various mitigation proposals have been proposed. LoRA is usually used in fine-tuning LLMs and a good entry point to insert privacy-enhancing modules. In this ongoing research, we introduce PSY, a Posterior Sampling based PrivacY enhancer that can be used in LoRA. We propose a simple yet effective realization of PSY using posterior sampling, which effectively prevents privacy leakage from intermediate information and, in turn, preserves the privacy of data owners. We evaluate LoRA extended with PSY against state-of-the-art membership inference and data extraction attacks. The experiments are executed on three different LLM architectures fine-tuned on three datasets with LoRA. In contrast to the commonly used differential privacy method, we find that our proposed modification consistently reduces the attack success rate. Meanwhile, our method has almost no negative impact on model fine-tuning or final performance. Most importantly, PSY reveals a promising path toward privacy enhancement with latent space extensions.



## **3. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

18 pages

**SubmitDate**: 2024-10-24    [abs](http://arxiv.org/abs/2410.18469v1) [paper-pdf](http://arxiv.org/pdf/2410.18469v1)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99% ASR on GPT-3.5 and 49% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM



## **4. Advancing NLP Security by Leveraging LLMs as Adversarial Engines**

cs.AI

5 pages

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18215v1) [paper-pdf](http://arxiv.org/pdf/2410.18215v1)

**Authors**: Sudarshan Srinivasan, Maria Mahbub, Amir Sadovnik

**Abstract**: This position paper proposes a novel approach to advancing NLP security by leveraging Large Language Models (LLMs) as engines for generating diverse adversarial attacks. Building upon recent work demonstrating LLMs' effectiveness in creating word-level adversarial examples, we argue for expanding this concept to encompass a broader range of attack types, including adversarial patches, universal perturbations, and targeted attacks. We posit that LLMs' sophisticated language understanding and generation capabilities can produce more effective, semantically coherent, and human-like adversarial examples across various domains and classifier architectures. This paradigm shift in adversarial NLP has far-reaching implications, potentially enhancing model robustness, uncovering new vulnerabilities, and driving innovation in defense mechanisms. By exploring this new frontier, we aim to contribute to the development of more secure, reliable, and trustworthy NLP systems for critical applications.



## **5. Towards Understanding the Fragility of Multilingual LLMs against Fine-Tuning Attacks**

cs.CL

14 pages, 6 figures, 7 tables

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.18210v1) [paper-pdf](http://arxiv.org/pdf/2410.18210v1)

**Authors**: Samuele Poppi, Zheng-Xin Yong, Yifei He, Bobbie Chern, Han Zhao, Aobo Yang, Jianfeng Chi

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked widespread concerns about their safety. Recent work demonstrates that safety alignment of LLMs can be easily removed by fine-tuning with a few adversarially chosen instruction-following examples, i.e., fine-tuning attacks. We take a further step to understand fine-tuning attacks in multilingual LLMs. We first discover cross-lingual generalization of fine-tuning attacks: using a few adversarially chosen instruction-following examples in one language, multilingual LLMs can also be easily compromised (e.g., multilingual LLMs fail to refuse harmful prompts in other languages). Motivated by this finding, we hypothesize that safety-related information is language-agnostic and propose a new method termed Safety Information Localization (SIL) to identify the safety-related information in the model parameter space. Through SIL, we validate this hypothesis and find that only changing 20% of weight parameters in fine-tuning attacks can break safety alignment across all languages. Furthermore, we provide evidence to the alternative pathways hypothesis for why freezing safety-related parameters does not prevent fine-tuning attacks, and we demonstrate that our attack vector can still jailbreak LLMs adapted to new languages.



## **6. Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02916v2) [paper-pdf](http://arxiv.org/pdf/2410.02916v2)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern of large language models (LLMs) in their open deployment. To this end, safeguard methods aim to enforce the ethical and responsible use of LLMs through safety alignment or guardrail mechanisms. However, we found that the malicious attackers could exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a new denial-of-service (DoS) attack on LLMs. Specifically, by software or phishing attacks on user client software, attackers insert a short, seemingly innocuous adversarial prompt into to user prompt templates in configuration files; thus, this prompt appears in final user requests without visibility in the user interface and is not trivial to identify. By designing an optimization process that utilizes gradient and attention information, our attack can automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97\% of user requests on Llama Guard 3. The attack presents a new dimension of evaluating LLM safeguards focusing on false positives, fundamentally different from the classic jailbreak.



## **7. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.02240v4) [paper-pdf](http://arxiv.org/pdf/2410.02240v4)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **8. Guide for Defense (G4D): Dynamic Guidance for Robust and Balanced Defense in Large Language Models**

cs.AI

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.17922v1) [paper-pdf](http://arxiv.org/pdf/2410.17922v1)

**Authors**: He Cao, Weidi Luo, Yu Wang, Zijing Liu, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.



## **9. IBGP: Imperfect Byzantine Generals Problem for Zero-Shot Robustness in Communicative Multi-Agent Systems**

cs.MA

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2410.16237v2) [paper-pdf](http://arxiv.org/pdf/2410.16237v2)

**Authors**: Yihuan Mao, Yipeng Kang, Peilun Li, Ning Zhang, Wei Xu, Chongjie Zhang

**Abstract**: As large language model (LLM) agents increasingly integrate into our infrastructure, their robust coordination and message synchronization become vital. The Byzantine Generals Problem (BGP) is a critical model for constructing resilient multi-agent systems (MAS) under adversarial attacks. It describes a scenario where malicious agents with unknown identities exist in the system-situations that, in our context, could result from LLM agents' hallucinations or external attacks. In BGP, the objective of the entire system is to reach a consensus on the action to be taken. Traditional BGP requires global consensus among all agents; however, in practical scenarios, global consensus is not always necessary and can even be inefficient. Therefore, there is a pressing need to explore a refined version of BGP that aligns with the local coordination patterns observed in MAS. We refer to this refined version as Imperfect BGP (IBGP) in our research, aiming to address this discrepancy. To tackle this issue, we propose a framework that leverages consensus protocols within general MAS settings, providing provable resilience against communication attacks and adaptability to changing environments, as validated by empirical results. Additionally, we present a case study in a sensor network environment to illustrate the practical application of our protocol.



## **10. ConfusedPilot: Confused Deputy Risks in RAG-based LLMs**

cs.CR

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2408.04870v5) [paper-pdf](http://arxiv.org/pdf/2408.04870v5)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.



## **11. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

cs.CL

14 pages, 7 figures

**SubmitDate**: 2024-10-23    [abs](http://arxiv.org/abs/2402.10601v2) [paper-pdf](http://arxiv.org/pdf/2402.10601v2)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Chitta Baral

**Abstract**: Recent advancements in the safety of Large Language Models (LLMs) have primarily focused on mitigating attacks crafted in natural language or in common encryption techniques like Base64. However, new models which often possess better reasoning capabilities, open the door to new attack vectors that were previously non-existent in older models. This seems counter-intuitive at first glance, but these advanced models can decipher more complex cryptic queries that previous models could not, making them susceptible to attacks using such prompts. To exploit this vulnerability, we propose Attacks using Custom Encryptions (ACE), a novel method to jailbreak LLMs by leveraging custom encryption schemes. We evaluate the effectiveness of ACE on four state-of-the-art LLMs, achieving Attack Success Rates (ASR) of up to 66% on close-source models and 88% on open-source models. Building upon this, we introduce Layered Attacks using Custom Encryptions (LACE), which employs multiple layers of encryption through our custom ciphers to further enhance the ASR. Our findings demonstrate that LACE significantly enhances the ability to jailbreak LLMs, increasing the ASR of GPT-4o from 40% to 78%, a 38% improvement. Our results highlight that the advanced capabilities of LLMs introduce unforeseen vulnerabilities to complex attacks. Specifically complex and layered ciphers increase the chance of jailbreaking.



## **12. Learning to Poison Large Language Models During Instruction Tuning**

cs.LG

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2402.13459v2) [paper-pdf](http://arxiv.org/pdf/2402.13459v2)

**Authors**: Yao Qiang, Xiangyu Zhou, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during the instruction tuning of LLMs and emphasizes the necessity of safeguarding LLMs against data poisoning attacks.



## **13. Context-aware Prompt Tuning: Advancing In-Context Learning with Adversarial Methods**

cs.CL

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17222v1) [paper-pdf](http://arxiv.org/pdf/2410.17222v1)

**Authors**: Tsachi Blau, Moshe Kimhi, Yonatan Belinkov, Alexander Bronstein, Chaim Baskin

**Abstract**: Fine-tuning Large Language Models (LLMs) typically involves updating at least a few billions of parameters. A more parameter-efficient approach is Prompt Tuning (PT), which updates only a few learnable tokens, and differently, In-Context Learning (ICL) adapts the model to a new task by simply including examples in the input without any training. When applying optimization-based methods, such as fine-tuning and PT for few-shot learning, the model is specifically adapted to the small set of training examples, whereas ICL leaves the model unchanged. This distinction makes traditional learning methods more prone to overfitting; in contrast, ICL is less sensitive to the few-shot scenario. While ICL is not prone to overfitting, it does not fully extract the information that exists in the training examples. This work introduces Context-aware Prompt Tuning (CPT), a method inspired by ICL, PT, and adversarial attacks. We build on the ICL strategy of concatenating examples before the input, but we extend this by PT-like learning, refining the context embedding through iterative optimization to extract deeper insights from the training examples. We carefully modify specific context tokens, considering the unique structure of input and output formats. Inspired by adversarial attacks, we adjust the input based on the labels present in the context, focusing on minimizing, rather than maximizing, the loss. Moreover, we apply a projected gradient descent algorithm to keep token embeddings close to their original values, under the assumption that the user-provided data is inherently valuable. Our method has been shown to achieve superior accuracy across multiple classification tasks using various LLM models.



## **14. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

cs.CR

Accepted by Expert Systems With Applications

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2404.18816v3) [paper-pdf](http://arxiv.org/pdf/2404.18816v3)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous learning-based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing learning-based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Then, using our carefully crafted multi-view prompt templates, it guides the LLM to generate function descriptions and behavioral summaries for each view, enabling deep semantic analysis of the views. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the human-readable diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline methods. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.



## **15. Arabic Dataset for LLM Safeguard Evaluation**

cs.CL

17 pages, 6 figures, 10 tables

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.17040v1) [paper-pdf](http://arxiv.org/pdf/2410.17040v1)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.



## **16. Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations**

cs.CL

Findings of EMNLP Camera-ready version

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2404.13948v2) [paper-pdf](http://arxiv.org/pdf/2404.13948v2)

**Authors**: Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, Jong C. Park

**Abstract**: The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications. Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors. In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (\textit{GARAG}), which targets these aspects. Specifically, GARAG is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our \textit{GARAG} to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that GARAG consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.



## **17. Breaking ReAct Agents: Foot-in-the-Door Attack Will Get You In**

cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.16950v1) [paper-pdf](http://arxiv.org/pdf/2410.16950v1)

**Authors**: Itay Nakash, George Kour, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Following the advancement of large language models (LLMs), the development of LLM-based autonomous agents has become increasingly prevalent. As a result, the need to understand the security vulnerabilities of these agents has become a critical task. We examine how ReAct agents can be exploited using a straightforward yet effective method we refer to as the foot-in-the-door attack. Our experiments show that indirect prompt injection attacks, prompted by harmless and unrelated requests (such as basic calculations) can significantly increase the likelihood of the agent performing subsequent malicious actions. Our results show that once a ReAct agents thought includes a specific tool or action, the likelihood of executing this tool in the subsequent steps increases significantly, as the agent seldom re-evaluates its actions. Consequently, even random, harmless requests can establish a foot-in-the-door, allowing an attacker to embed malicious instructions into the agents thought process, making it more susceptible to harmful directives. To mitigate this vulnerability, we propose implementing a simple reflection mechanism that prompts the agent to reassess the safety of its actions during execution, which can help reduce the success of such attacks.



## **18. RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process**

cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.08660v2) [paper-pdf](http://arxiv.org/pdf/2410.08660v2)

**Authors**: Peiran Wang, Xiaogeng Liu, Chaowei Xiao

**Abstract**: In this study, we introduce RePD, an innovative attack Retrieval-based Prompt Decomposition framework designed to mitigate the risk of jailbreak attacks on large language models (LLMs). Despite rigorous pretraining and finetuning focused on ethical alignment, LLMs are still susceptible to jailbreak exploits. RePD operates on a one-shot learning model, wherein it accesses a database of pre-collected jailbreak prompt templates to identify and decompose harmful inquiries embedded within user prompts. This process involves integrating the decomposition of the jailbreak prompt into the user's original query into a one-shot learning example to effectively teach the LLM to discern and separate malicious components. Consequently, the LLM is equipped to first neutralize any potentially harmful elements before addressing the user's prompt in a manner that aligns with its ethical guidelines. RePD is versatile and compatible with a variety of open-source LLMs acting as agents. Through comprehensive experimentation with both harmful and benign prompts, we have demonstrated the efficacy of our proposed RePD in enhancing the resilience of LLMs against jailbreak attacks, without compromising their performance in responding to typical user requests.



## **19. $\textit{MMJ-Bench}$: A Comprehensive Study on Jailbreak Attacks and Defenses for Multimodal Large Language Models**

cs.CR

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2408.08464v4) [paper-pdf](http://arxiv.org/pdf/2408.08464v4)

**Authors**: Fenghua Weng, Yue Xu, Chengyan Fu, Wenjie Wang

**Abstract**: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Multimodal Large Language Models (MLLMs), have shown exceptional performance in many real-world tasks. However, MLLMs face significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety alignment to elicit harmful responses. The threat of jailbreak attacks on MLLMs arises from both the inherent vulnerabilities of LLMs and the multiple information channels that MLLMs process. While various attacks and defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense techniques for MLLMs. Through extensive experiments, we assess the effectiveness of various attack methods against SoTA MLLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the first public-available benchmark for MLLM jailbreak research. We also demonstrate several insightful findings that highlights directions for future studies.



## **20. Imprompter: Tricking LLM Agents into Improper Tool Use**

cs.CR

website: https://imprompter.ai code:  https://github.com/Reapor-Yurnero/imprompter v2 changelog: add new results to  Table 3, correct several typos

**SubmitDate**: 2024-10-22    [abs](http://arxiv.org/abs/2410.14923v2) [paper-pdf](http://arxiv.org/pdf/2410.14923v2)

**Authors**: Xiaohan Fu, Shuheng Li, Zihan Wang, Yihao Liu, Rajesh K. Gupta, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Model (LLM) Agents are an emerging computing paradigm that blends generative machine learning with tools such as code interpreters, web browsing, email, and more generally, external resources. These agent-based systems represent an emerging shift in personal computing. We contribute to the security foundations of agent-based systems and surface a new class of automatically computed obfuscated adversarial prompt attacks that violate the confidentiality and integrity of user resources connected to an LLM agent. We show how prompt optimization techniques can find such prompts automatically given the weights of a model. We demonstrate that such attacks transfer to production-level agents. For example, we show an information exfiltration attack on Mistral's LeChat agent that analyzes a user's conversation, picks out personally identifiable information, and formats it into a valid markdown command that results in leaking that data to the attacker's server. This attack shows a nearly 80% success rate in an end-to-end evaluation. We conduct a range of experiments to characterize the efficacy of these attacks and find that they reliably work on emerging agent-based systems like Mistral's LeChat, ChatGLM, and Meta's Llama. These attacks are multimodal, and we show variants in the text-only and image domains.



## **21. Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis**

cs.CR

15 pages, 11 figures

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16527v1) [paper-pdf](http://arxiv.org/pdf/2410.16527v1)

**Authors**: Jonathan Brokman, Omer Hofman, Oren Rachmil, Inderjeet Singh, Rathina Sabapathy, Aishvariya Priya, Vikas Pahuja, Amit Giloni, Roman Vainshtein, Hisashi Kojima

**Abstract**: This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.



## **22. Refusal-Trained LLMs Are Easily Jailbroken As Browser Agents**

cs.CR

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.13886v2) [paper-pdf](http://arxiv.org/pdf/2410.13886v2)

**Authors**: Priyanshu Kumar, Elaine Lau, Saranya Vijayakumar, Tu Trinh, Scale Red Team, Elaine Chang, Vaughn Robinson, Sean Hendryx, Shuyan Zhou, Matt Fredrikson, Summer Yue, Zifan Wang

**Abstract**: For safety reasons, large language models (LLMs) are trained to refuse harmful user instructions, such as assisting dangerous activities. We study an open question in this work: does the desired safety refusal, typically enforced in chat contexts, generalize to non-chat and agentic use cases? Unlike chatbots, LLM agents equipped with general-purpose tools, such as web browsers and mobile devices, can directly influence the real world, making it even more crucial to refuse harmful instructions. In this work, we primarily focus on red-teaming browser agents, LLMs that manipulate information via web browsers. To this end, we introduce Browser Agent Red teaming Toolkit (BrowserART), a comprehensive test suite designed specifically for red-teaming browser agents. BrowserART is consist of 100 diverse browser-related harmful behaviors (including original behaviors and ones sourced from HarmBench [Mazeika et al., 2024] and AirBench 2024 [Zeng et al., 2024b]) across both synthetic and real websites. Our empirical study on state-of-the-art browser agents reveals that, while the backbone LLM refuses harmful instructions as a chatbot, the corresponding agent does not. Moreover, attack methods designed to jailbreak refusal-trained LLMs in the chat settings transfer effectively to browser agents. With human rewrites, GPT-4o and o1-preview-based browser agents attempted 98 and 63 harmful behaviors (out of 100), respectively. We publicly release BrowserART and call on LLM developers, policymakers, and agent developers to collaborate on improving agent safety



## **23. A Realistic Threat Model for Large Language Model Jailbreaks**

cs.LG

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16222v1) [paper-pdf](http://arxiv.org/pdf/2410.16222v1)

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. In their original settings, these methods all largely succeed in coercing the target output, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model combines constraints in perplexity, measuring how far a jailbreak deviates from natural text, and computational budget, in total FLOPs. For the former, we build an N-gram model on 1T tokens, which, in contrast to model-based perplexity, allows for an LLM-agnostic and inherently interpretable evaluation. We adapt popular attacks to this new, realistic threat model, with which we, for the first time, benchmark these attacks on equal footing. After a rigorous comparison, we not only find attack success rates against safety-tuned modern models to be lower than previously presented but also find that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent N-grams, either selecting N-grams absent from real-world text or rare ones, e.g. specific to code datasets.



## **24. Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2409.18169v3) [paper-pdf](http://arxiv.org/pdf/2409.18169v3)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Recent research demonstrates that the nascent fine-tuning-as-a-service business model exposes serious safety concerns -- fine-tuning over a few harmful data uploaded by the users can compromise the safety alignment of the model. The attack, known as harmful fine-tuning, has raised a broad research interest among the community. However, as the attack is still new, \textbf{we observe from our miserable submission experience that there are general misunderstandings within the research community.} We in this paper aim to clear some common concerns for the attack setting, and formally establish the research problem. Specifically, we first present the threat model of the problem, and introduce the harmful fine-tuning attack and its variants. Then we systematically survey the existing literature on attacks/defenses/mechanical analysis of the problem. Finally, we outline future research directions that might contribute to the development of the field. Additionally, we present a list of questions of interest, which might be useful to refer to when reviewers in the peer review process question the realism of the experiment/attack/defense setting. A curated list of relevant papers is maintained and made accessible at: \url{https://github.com/git-disl/awesome_LLM-harmful-fine-tuning-papers}.



## **25. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

cs.CL

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16155v1) [paper-pdf](http://arxiv.org/pdf/2410.16155v1)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.



## **26. Extracting Spatiotemporal Data from Gradients with Large Language Models**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2407.08529

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.16121v1) [paper-pdf](http://arxiv.org/pdf/2410.16121v1)

**Authors**: Lele Zheng, Yang Cao, Renhe Jiang, Kenjiro Taura, Yulong Shen, Sheng Li, Masatoshi Yoshikawa

**Abstract**: Recent works show that sensitive user data can be reconstructed from gradient updates, breaking the key privacy promise of federated learning. While success was demonstrated primarily on image data, these methods do not directly transfer to other domains, such as spatiotemporal data. To understand privacy risks in spatiotemporal federated learning, we first propose Spatiotemporal Gradient Inversion Attack (ST-GIA), a gradient attack algorithm tailored to spatiotemporal data that successfully reconstructs the original location from gradients. Furthermore, the absence of priors in attacks on spatiotemporal data has hindered the accurate reconstruction of real client data. To address this limitation, we propose ST-GIA+, which utilizes an auxiliary language model to guide the search for potential locations, thereby successfully reconstructing the original data from gradients. In addition, we design an adaptive defense strategy to mitigate gradient inversion attacks in spatiotemporal federated learning. By dynamically adjusting the perturbation levels, we can offer tailored protection for varying rounds of training data, thereby achieving a better trade-off between privacy and utility than current state-of-the-art methods. Through intensive experimental analysis on three real-world datasets, we reveal that the proposed defense strategy can well preserve the utility of spatiotemporal federated learning with effective security protection.



## **27. NetSafe: Exploring the Topological Safety of Multi-agent Networks**

cs.MA

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15686v1) [paper-pdf](http://arxiv.org/pdf/2410.15686v1)

**Authors**: Miao Yu, Shilong Wang, Guibin Zhang, Junyuan Mao, Chenlong Yin, Qijiong Liu, Qingsong Wen, Kun Wang, Yang Wang

**Abstract**: Large language models (LLMs) have empowered nodes within multi-agent networks with intelligence, showing growing applications in both academia and industry. However, how to prevent these networks from generating malicious information remains unexplored with previous research on single LLM's safety be challenging to transfer. In this paper, we focus on the safety of multi-agent networks from a topological perspective, investigating which topological properties contribute to safer networks. To this end, we propose a general framework, NetSafe along with an iterative RelCom interaction to unify existing diverse LLM-based agent frameworks, laying the foundation for generalized topological safety research. We identify several critical phenomena when multi-agent networks are exposed to attacks involving misinformation, bias, and harmful information, termed as Agent Hallucination and Aggregation Safety. Furthermore, we find that highly connected networks are more susceptible to the spread of adversarial attacks, with task performance in a Star Graph Topology decreasing by 29.7%. Besides, our proposed static metrics aligned more closely with real-world dynamic evaluations than traditional graph-theoretic metrics, indicating that networks with greater average distances from attackers exhibit enhanced safety. In conclusion, our work introduces a new topological perspective on the safety of LLM-based multi-agent networks and discovers several unreported phenomena, paving the way for future research to explore the safety of such networks.



## **28. Boosting Jailbreak Transferability for Large Language Models**

cs.AI

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15645v1) [paper-pdf](http://arxiv.org/pdf/2410.15645v1)

**Authors**: Hanqing Liu, Lifeng Zhou, Huanqian Yan

**Abstract**: Large language models have drawn significant attention to the challenge of safe alignment, especially regarding jailbreak attacks that circumvent security measures to produce harmful content. To address the limitations of existing methods like GCG, which perform well in single-model attacks but lack transferability, we propose several enhancements, including a scenario induction template, optimized suffix selection, and the integration of re-suffix attack mechanism to reduce inconsistent outputs. Our approach has shown superior performance in extensive experiments across various benchmarks, achieving nearly 100% success rates in both attack execution and transferability. Notably, our method has won the online first place in the AISG-hosted Global Challenge for Safe and Secure LLMs.



## **29. SMILES-Prompting: A Novel Approach to LLM Jailbreak Attacks in Chemical Synthesis**

cs.CL

**SubmitDate**: 2024-10-21    [abs](http://arxiv.org/abs/2410.15641v1) [paper-pdf](http://arxiv.org/pdf/2410.15641v1)

**Authors**: Aidan Wong, He Cao, Zijing Liu, Yu Li

**Abstract**: The increasing integration of large language models (LLMs) across various fields has heightened concerns about their potential to propagate dangerous information. This paper specifically explores the security vulnerabilities of LLMs within the field of chemistry, particularly their capacity to provide instructions for synthesizing hazardous substances. We evaluate the effectiveness of several prompt injection attack methods, including red-teaming, explicit prompting, and implicit prompting. Additionally, we introduce a novel attack technique named SMILES-prompting, which uses the Simplified Molecular-Input Line-Entry System (SMILES) to reference chemical substances. Our findings reveal that SMILES-prompting can effectively bypass current safety mechanisms. These findings highlight the urgent need for enhanced domain-specific safeguards in LLMs to prevent misuse and improve their potential for positive social impact.



## **30. Revisit, Extend, and Enhance Hessian-Free Influence Functions**

cs.LG

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2405.17490v2) [paper-pdf](http://arxiv.org/pdf/2405.17490v2)

**Authors**: Ziao Yang, Han Yue, Jian Chen, Hongfu Liu

**Abstract**: Influence functions serve as crucial tools for assessing sample influence in model interpretation, subset training set selection, noisy label detection, and more. By employing the first-order Taylor extension, influence functions can estimate sample influence without the need for expensive model retraining. However, applying influence functions directly to deep models presents challenges, primarily due to the non-convex nature of the loss function and the large size of model parameters. This difficulty not only makes computing the inverse of the Hessian matrix costly but also renders it non-existent in some cases. Various approaches, including matrix decomposition, have been explored to expedite and approximate the inversion of the Hessian matrix, with the aim of making influence functions applicable to deep models. In this paper, we revisit a specific, albeit naive, yet effective approximation method known as TracIn. This method substitutes the inverse of the Hessian matrix with an identity matrix. We provide deeper insights into why this simple approximation method performs well. Furthermore, we extend its applications beyond measuring model utility to include considerations of fairness and robustness. Finally, we enhance TracIn through an ensemble strategy. To validate its effectiveness, we conduct experiments on synthetic data and extensive evaluations on noisy label detection, sample selection for large language model fine-tuning, and defense against adversarial attacks.



## **31. The Best Defense is a Good Offense: Countering LLM-Powered Cyberattacks**

cs.CR

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.15396v1) [paper-pdf](http://arxiv.org/pdf/2410.15396v1)

**Authors**: Daniel Ayzenshteyn, Roy Weiss, Yisroel Mirsky

**Abstract**: As large language models (LLMs) continue to evolve, their potential use in automating cyberattacks becomes increasingly likely. With capabilities such as reconnaissance, exploitation, and command execution, LLMs could soon become integral to autonomous cyber agents, capable of launching highly sophisticated attacks. In this paper, we introduce novel defense strategies that exploit the inherent vulnerabilities of attacking LLMs. By targeting weaknesses such as biases, trust in input, memory limitations, and their tunnel-vision approach to problem-solving, we develop techniques to mislead, delay, or neutralize these autonomous agents. We evaluate our defenses under black-box conditions, starting with single prompt-response scenarios and progressing to real-world tests using custom-built CTF machines. Our results show defense success rates of up to 90\%, demonstrating the effectiveness of turning LLM vulnerabilities into defensive strategies against LLM-driven cyber threats.



## **32. Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models**

cs.LG

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.15362v1) [paper-pdf](http://arxiv.org/pdf/2410.15362v1)

**Authors**: Xiao Li, Zhuhong Li, Qiongxiu Li, Bingze Lee, Jinghao Cui, Xiaolin Hu

**Abstract**: Aligned Large Language Models (LLMs) have demonstrated remarkable performance across various tasks. However, LLMs remain susceptible to jailbreak adversarial attacks, where adversaries manipulate prompts to elicit malicious responses that aligned LLMs should have avoided. Identifying these vulnerabilities is crucial for understanding the inherent weaknesses of LLMs and preventing their potential misuse. One pioneering work in jailbreaking is the GCG attack, a discrete token optimization algorithm that seeks to find a suffix capable of jailbreaking aligned LLMs. Despite the success of GCG, we find it suboptimal, requiring significantly large computational costs, and the achieved jailbreaking performance is limited. In this work, we propose Faster-GCG, an efficient adversarial jailbreak method by delving deep into the design of GCG. Experiments demonstrate that Faster-GCG can surpass the original GCG with only 1/10 of the computational cost, achieving significantly higher attack success rates on various open-source aligned LLMs. In addition, We demonstrate that Faster-GCG exhibits improved attack transferability when testing on closed-sourced LLMs such as ChatGPT.



## **33. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

cs.CR

**SubmitDate**: 2024-10-20    [abs](http://arxiv.org/abs/2410.15236v1) [paper-pdf](http://arxiv.org/pdf/2410.15236v1)

**Authors**: Benji Peng, Ziqian Bi, Qian Niu, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.



## **34. Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks**

cs.CR

17 pages, 1 figure

**SubmitDate**: 2024-10-19    [abs](http://arxiv.org/abs/2409.08087v2) [paper-pdf](http://arxiv.org/pdf/2409.08087v2)

**Authors**: Benji Peng, Keyu Chen, Ming Li, Pohsun Feng, Ziqian Bi, Junyu Liu, Qian Niu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across various fields, yet their increasing use raises critical security concerns. This article reviews recent literature addressing key issues in LLM security, with a focus on accuracy, bias, content detection, and vulnerability to attacks. Issues related to inaccurate or misleading outputs from LLMs is discussed, with emphasis on the implementation from fact-checking methodologies to enhance response reliability. Inherent biases within LLMs are critically examined through diverse evaluation techniques, including controlled input studies and red teaming exercises. A comprehensive analysis of bias mitigation strategies is presented, including approaches from pre-processing interventions to in-training adjustments and post-processing refinements. The article also probes the complexity of distinguishing LLM-generated content from human-produced text, introducing detection mechanisms like DetectGPT and watermarking techniques while noting the limitations of machine learning enabled classifiers under intricate circumstances. Moreover, LLM vulnerabilities, including jailbreak attacks and prompt injection exploits, are analyzed by looking into different case studies and large-scale competitions like HackAPrompt. This review is concluded by retrospecting defense mechanisms to safeguard LLMs, accentuating the need for more extensive research into the LLM security field.



## **35. Multi-round jailbreak attack on large language models**

cs.CL

It is not fully completed

**SubmitDate**: 2024-10-19    [abs](http://arxiv.org/abs/2410.11533v2) [paper-pdf](http://arxiv.org/pdf/2410.11533v2)

**Authors**: Yihua Zhou, Xiaochuan Shi

**Abstract**: Ensuring the safety and alignment of large language models (LLMs) with human values is crucial for generating responses that are beneficial to humanity. While LLMs have the capability to identify and avoid harmful queries, they remain vulnerable to "jailbreak" attacks, where carefully crafted prompts can induce the generation of toxic content. Traditional single-round jailbreak attacks, such as GCG and AutoDAN, do not alter the sensitive words in the dangerous prompts. Although they can temporarily bypass the model's safeguards through prompt engineering, their success rate drops significantly as the LLM is further fine-tuned, and they cannot effectively circumvent static rule-based filters that remove the hazardous vocabulary.   In this study, to better understand jailbreak attacks, we introduce a multi-round jailbreak approach. This method can rewrite the dangerous prompts, decomposing them into a series of less harmful sub-questions to bypass the LLM's safety checks. We first use the LLM to perform a decomposition task, breaking down a set of natural language questions into a sequence of progressive sub-questions, which are then used to fine-tune the Llama3-8B model, enabling it to decompose hazardous prompts. The fine-tuned model is then used to break down the problematic prompt, and the resulting sub-questions are sequentially asked to the victim model. If the victim model rejects a sub-question, a new decomposition is generated, and the process is repeated until the final objective is achieved. Our experimental results show a 94\% success rate on the llama2-7B and demonstrate the effectiveness of this approach in circumventing static rule-based filters.



## **36. ASTPrompter: Weakly Supervised Automated Language Model Red-Teaming to Identify Likely Toxic Prompts**

cs.CL

8 pages, 8 pages of appendix, 2 tables, 3 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2407.09447v2) [paper-pdf](http://arxiv.org/pdf/2407.09447v2)

**Authors**: Amelia F. Hardy, Houjun Liu, Bernard Lange, Mykel J. Kochenderfer

**Abstract**: Typical schemes for the automated red-teaming of large language models (LLMs) focus on discovering prompts that trigger a frozen language model (the defender) to generate toxic text. This often results in the prompting model (the adversary) producing text that is unintelligible and unlikely to arise. Here, we propose a reinforcement learning formulation of the LLM red-teaming task that allows us to discover prompts that both (1) trigger toxic outputs from a frozen defender and (2) have low perplexity as scored by that defender. We argue these cases are the most pertinent in a red-teaming setting because they are likely to arise during normal use of the defender model. We solve this formulation through a novel online and weakly supervised variant of Identity Preference Optimization (IPO) on GPT-2, GPT-2 XL, and TinyLlama defenders. We demonstrate that our policy is capable of generating likely (low-perplexity) prompts that also trigger toxicity from all of these architectures. Furthermore, we show that this policy outperforms baselines by producing attacks that are occur with higher probability and are more effective. Finally, we discuss our findings and the observed trade-offs between likelihood vs toxicity. Source code for this project is available for this project at: https://github.com/sisl/ASTPrompter/.



## **37. Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs**

cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.16327v1) [paper-pdf](http://arxiv.org/pdf/2410.16327v1)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Zejian Chen, Litian Zhang, Zheng Liu, Lirong Qiu, Xi Zhang

**Abstract**: Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.



## **38. When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs**

cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14569v1) [paper-pdf](http://arxiv.org/pdf/2410.14569v1)

**Authors**: Hanna Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin, Kimin Lee

**Abstract**: Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.



## **39. BlackDAN: A Black-Box Multi-Objective Approach for Effective and Contextual Jailbreaking of Large Language Models**

cs.CR

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.09804v2) [paper-pdf](http://arxiv.org/pdf/2410.09804v2)

**Authors**: Xinyuan Wang, Victor Shea-Jay Huang, Renmiao Chen, Hao Wang, Chengwei Pan, Lei Sha, Minlie Huang

**Abstract**: While large language models (LLMs) exhibit remarkable capabilities across various tasks, they encounter potential security risks such as jailbreak attacks, which exploit vulnerabilities to bypass security measures and generate harmful outputs. Existing jailbreak strategies mainly focus on maximizing attack success rate (ASR), frequently neglecting other critical factors, including the relevance of the jailbreak response to the query and the level of stealthiness. This narrow focus on single objectives can result in ineffective attacks that either lack contextual relevance or are easily recognizable. In this work, we introduce BlackDAN, an innovative black-box attack framework with multi-objective optimization, aiming to generate high-quality prompts that effectively facilitate jailbreaking while maintaining contextual relevance and minimizing detectability. BlackDAN leverages Multiobjective Evolutionary Algorithms (MOEAs), specifically the NSGA-II algorithm, to optimize jailbreaks across multiple objectives including ASR, stealthiness, and semantic relevance. By integrating mechanisms like mutation, crossover, and Pareto-dominance, BlackDAN provides a transparent and interpretable process for generating jailbreaks. Furthermore, the framework allows customization based on user preferences, enabling the selection of prompts that balance harmfulness, relevance, and other factors. Experimental results demonstrate that BlackDAN outperforms traditional single-objective methods, yielding higher success rates and improved robustness across various LLMs and multimodal LLMs, while ensuring jailbreak responses are both relevant and less detectable.



## **40. Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented Generation of Large Language Models**

cs.CR

12 pages, 5 figures

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14479v1) [paper-pdf](http://arxiv.org/pdf/2410.14479v1)

**Authors**: Cody Clop, Yannick Teglia

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating coherent text but remain limited by the static nature of their training data. Retrieval Augmented Generation (RAG) addresses this issue by combining LLMs with up-to-date information retrieval, but also expand the attack surface of the system. This paper investigates prompt injection attacks on RAG, focusing on malicious objectives beyond misinformation, such as inserting harmful links, promoting unauthorized services, and initiating denial-of-service behaviors. We build upon existing corpus poisoning techniques and propose a novel backdoor attack aimed at the fine-tuning process of the dense retriever component. Our experiments reveal that corpus poisoning can achieve significant attack success rates through the injection of a small number of compromised documents into the retriever corpus. In contrast, backdoor attacks demonstrate even higher success rates but necessitate a more complex setup, as the victim must fine-tune the retriever using the attacker poisoned dataset.



## **41. Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation**

cs.CL

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.14425v1) [paper-pdf](http://arxiv.org/pdf/2410.14425v1)

**Authors**: Shuai Zhao, Xiaobao Wu, Cong-Duy Nguyen, Meihuizi Jia, Yichao Feng, Luu Anh Tuan

**Abstract**: Parameter-efficient fine-tuning (PEFT) can bridge the gap between large language models (LLMs) and downstream tasks. However, PEFT has been proven vulnerable to malicious attacks. Research indicates that poisoned LLMs, even after PEFT, retain the capability to activate internalized backdoors when input samples contain predefined triggers. In this paper, we introduce a novel weak-to-strong unlearning algorithm to defend against backdoor attacks based on feature alignment knowledge distillation, named W2SDefense. Specifically, we first train a small-scale language model through full-parameter fine-tuning to serve as the clean teacher model. Then, this teacher model guides the large-scale poisoned student model in unlearning the backdoor, leveraging PEFT. Theoretical analysis suggests that W2SDefense has the potential to enhance the student model's ability to unlearn backdoor features, preventing the activation of the backdoor. We conduct experiments on text classification tasks involving three state-of-the-art language models and three different backdoor attack algorithms. Our empirical results demonstrate the outstanding performance of W2SDefense in defending against backdoor attacks without compromising model performance.



## **42. VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment**

cs.CV

EMNLP 2024 Main Conference camera-ready version (fixed small typos).  This article supersedes arXiv:2312.10665

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.09421v2) [paper-pdf](http://arxiv.org/pdf/2410.09421v2)

**Authors**: Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, Qi Liu

**Abstract**: As large vision-language models (LVLMs) evolve rapidly, the demand for high-quality and diverse data to align these models becomes increasingly crucial. However, the creation of such data with human supervision proves costly and time-intensive. In this paper, we investigate the efficacy of AI feedback to scale supervision for aligning LVLMs. We introduce VLFeedback, the first large-scale vision-language feedback dataset, comprising over 82K multi-modal instructions and comprehensive rationales generated by off-the-shelf models without human annotations. To evaluate the effectiveness of AI feedback for vision-language alignment, we train Silkie, an LVLM fine-tuned via direct preference optimization on VLFeedback. Silkie showcases exceptional performance regarding helpfulness, visual faithfulness, and safety metrics. It outperforms its base model by 6.9\% and 9.5\% in perception and cognition tasks, reduces hallucination issues on MMHal-Bench, and exhibits enhanced resilience against red-teaming attacks. Furthermore, our analysis underscores the advantage of AI feedback, particularly in fostering preference diversity to deliver more comprehensive improvements. Our dataset, training code and models are available at https://vlf-silkie.github.io.



## **43. DomainLynx: Leveraging Large Language Models for Enhanced Domain Squatting Detection**

cs.CR

Accepted for publication at IEEE CCNC 2025

**SubmitDate**: 2024-10-18    [abs](http://arxiv.org/abs/2410.02095v2) [paper-pdf](http://arxiv.org/pdf/2410.02095v2)

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Domain squatting poses a significant threat to Internet security, with attackers employing increasingly sophisticated techniques. This study introduces DomainLynx, an innovative compound AI system leveraging Large Language Models (LLMs) for enhanced domain squatting detection. Unlike existing methods focusing on predefined patterns for top-ranked domains, DomainLynx excels in identifying novel squatting techniques and protecting less prominent brands. The system's architecture integrates advanced data processing, intelligent domain pairing, and LLM-powered threat assessment. Crucially, DomainLynx incorporates specialized components that mitigate LLM hallucinations, ensuring reliable and context-aware detection. This approach enables efficient analysis of vast security data from diverse sources, including Certificate Transparency logs, Passive DNS records, and zone files. Evaluated on a curated dataset of 1,649 squatting domains, DomainLynx achieved 94.7\% accuracy using Llama-3-70B. In a month-long real-world test, it detected 34,359 squatting domains from 2.09 million new domains, outperforming baseline methods by 2.5 times. This research advances Internet security by providing a versatile, accurate, and adaptable tool for combating evolving domain squatting threats. DomainLynx's approach paves the way for more robust, AI-driven cybersecurity solutions, enhancing protection for a broader range of online entities and contributing to a safer digital ecosystem.



## **44. PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment**

cs.CL

28 pages

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13785v1) [paper-pdf](http://arxiv.org/pdf/2410.13785v1)

**Authors**: Zekun Moore Wang, Shawn Wang, Kang Zhu, Jiaheng Liu, Ke Xu, Jie Fu, Wangchunshu Zhou, Wenhao Huang

**Abstract**: Alignment of large language models (LLMs) involves training models on preference-contrastive output pairs to adjust their responses according to human preferences. To obtain such contrastive pairs, traditional methods like RLHF and RLAIF rely on limited contrasting patterns, such as varying model variants or decoding temperatures. This singularity leads to two issues: (1) alignment is not comprehensive; and thereby (2) models are susceptible to jailbreaking attacks. To address these issues, we investigate how to construct more comprehensive and diversified contrasting patterns to enhance preference data (RQ1) and verify the impact of the diversification of contrasting patterns on model alignment (RQ2). For RQ1, we propose PopAlign, a framework that integrates diversified contrasting patterns across the prompt, model, and pipeline levels, introducing six contrasting strategies that do not require additional feedback labeling procedures. Regarding RQ2, we conduct thorough experiments demonstrating that PopAlign significantly outperforms existing methods, leading to more comprehensive alignment.



## **45. Persistent Pre-Training Poisoning of LLMs**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13722v1) [paper-pdf](http://arxiv.org/pdf/2410.13722v1)

**Authors**: Yiming Zhang, Javier Rando, Ivan Evtimov, Jianfeng Chi, Eric Michael Smith, Nicholas Carlini, Florian Tramèr, Daphne Ippolito

**Abstract**: Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web. Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets. Our work evaluates for the first time whether language models can also be compromised during pre-training, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO). We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B). Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.



## **46. Jailbreaking LLM-Controlled Robots**

cs.RO

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13691v1) [paper-pdf](http://arxiv.org/pdf/2410.13691v1)

**Authors**: Alexander Robey, Zachary Ravichandran, Vijay Kumar, Hamed Hassani, George J. Pappas

**Abstract**: The recent introduction of large language models (LLMs) has revolutionized the field of robotics by enabling contextual reasoning and intuitive human-robot interaction in domains as varied as manipulation, locomotion, and self-driving vehicles. When viewed as a stand-alone technology, LLMs are known to be vulnerable to jailbreaking attacks, wherein malicious prompters elicit harmful text by bypassing LLM safety guardrails. To assess the risks of deploying LLMs in robotics, in this paper, we introduce RoboPAIR, the first algorithm designed to jailbreak LLM-controlled robots. Unlike existing, textual attacks on LLM chatbots, RoboPAIR elicits harmful physical actions from LLM-controlled robots, a phenomenon we experimentally demonstrate in three scenarios: (i) a white-box setting, wherein the attacker has full access to the NVIDIA Dolphins self-driving LLM, (ii) a gray-box setting, wherein the attacker has partial access to a Clearpath Robotics Jackal UGV robot equipped with a GPT-4o planner, and (iii) a black-box setting, wherein the attacker has only query access to the GPT-3.5-integrated Unitree Robotics Go2 robot dog. In each scenario and across three new datasets of harmful robotic actions, we demonstrate that RoboPAIR, as well as several static baselines, finds jailbreaks quickly and effectively, often achieving 100% attack success rates. Our results reveal, for the first time, that the risks of jailbroken LLMs extend far beyond text generation, given the distinct possibility that jailbroken robots could cause physical damage in the real world. Indeed, our results on the Unitree Go2 represent the first successful jailbreak of a deployed commercial robotic system. Addressing this emerging vulnerability is critical for ensuring the safe deployment of LLMs in robotics. Additional media is available at: https://robopair.org



## **47. Targeted Vaccine: Safety Alignment for Large Language Models against Harmful Fine-Tuning via Layer-wise Perturbation**

cs.LG

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.09760v2) [paper-pdf](http://arxiv.org/pdf/2410.09760v2)

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Li Shen

**Abstract**: Harmful fine-tuning attack poses a serious threat to the online fine-tuning service. Vaccine, a recent alignment-stage defense, applies uniform perturbation to all layers of embedding to make the model robust to the simulated embedding drift. However, applying layer-wise uniform perturbation may lead to excess perturbations for some particular safety-irrelevant layers, resulting in defense performance degradation and unnecessary memory consumption. To address this limitation, we propose Targeted Vaccine (T-Vaccine), a memory-efficient safety alignment method that applies perturbation to only selected layers of the model. T-Vaccine follows two core steps: First, it uses gradient norm as a statistical metric to identify the safety-critical layers. Second, instead of applying uniform perturbation across all layers, T-Vaccine only applies perturbation to the safety-critical layers while keeping other layers frozen during training. Results show that T-Vaccine outperforms Vaccine in terms of both defense effectiveness and resource efficiency. Comparison with other defense baselines, e.g., RepNoise and TAR also demonstrate the superiority of T-Vaccine. Notably, T-Vaccine is the first defense that can address harmful fine-tuning issues for a 7B pre-trained models trained on consumer GPUs with limited memory (e.g., RTX 4090). Our code is available at https://github.com/Lslland/T-Vaccine.



## **48. Breaking Chains: Unraveling the Links in Multi-Hop Knowledge Unlearning**

cs.CL

16 pages, 5 figures

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13274v1) [paper-pdf](http://arxiv.org/pdf/2410.13274v1)

**Authors**: Minseok Choi, ChaeHun Park, Dohyun Lee, Jaegul Choo

**Abstract**: Large language models (LLMs) serve as giant information stores, often including personal or copyrighted data, and retraining them from scratch is not a viable option. This has led to the development of various fast, approximate unlearning techniques to selectively remove knowledge from LLMs. Prior research has largely focused on minimizing the probabilities of specific token sequences by reversing the language modeling objective. However, these methods still leave LLMs vulnerable to adversarial attacks that exploit indirect references. In this work, we examine the limitations of current unlearning techniques in effectively erasing a particular type of indirect prompt: multi-hop queries. Our findings reveal that existing methods fail to completely remove multi-hop knowledge when one of the intermediate hops is unlearned. To address this issue, we propose MUNCH, a simple uncertainty-based approach that breaks down multi-hop queries into subquestions and leverages the uncertainty of the unlearned model in final decision-making. Empirical results demonstrate the effectiveness of our framework, and MUNCH can be easily integrated with existing unlearning techniques, making it a flexible and useful solution for enhancing unlearning processes.



## **49. FRAG: Toward Federated Vector Database Management for Collaborative and Secure Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13272v1) [paper-pdf](http://arxiv.org/pdf/2410.13272v1)

**Authors**: Dongfang Zhao

**Abstract**: This paper introduces \textit{Federated Retrieval-Augmented Generation (FRAG)}, a novel database management paradigm tailored for the growing needs of retrieval-augmented generation (RAG) systems, which are increasingly powered by large-language models (LLMs). FRAG enables mutually-distrusted parties to collaboratively perform Approximate $k$-Nearest Neighbor (ANN) searches on encrypted query vectors and encrypted data stored in distributed vector databases, all while ensuring that no party can gain any knowledge about the queries or data of others. Achieving this paradigm presents two key challenges: (i) ensuring strong security guarantees, such as Indistinguishability under Chosen-Plaintext Attack (IND-CPA), under practical assumptions (e.g., we avoid overly optimistic assumptions like non-collusion among parties); and (ii) maintaining performance overheads comparable to traditional, non-federated RAG systems. To address these challenges, FRAG employs a single-key homomorphic encryption protocol that simplifies key management across mutually-distrusted parties. Additionally, FRAG introduces a \textit{multiplicative caching} technique to efficiently encrypt floating-point numbers, significantly improving computational performance in large-scale federated environments. We provide a rigorous security proof using standard cryptographic reductions and demonstrate the practical scalability and efficiency of FRAG through extensive experiments on both benchmark and real-world datasets.



## **50. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

cs.CL

17 pages, 6 figures, 14 tables

**SubmitDate**: 2024-10-17    [abs](http://arxiv.org/abs/2410.13237v1) [paper-pdf](http://arxiv.org/pdf/2410.13237v1)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.



