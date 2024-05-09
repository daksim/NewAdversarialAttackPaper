# Latest Large Language Model Attack Papers
**update at 2024-05-09 23:51:11**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Locally Differentially Private In-Context Learning**

cs.CR

This paper was published at LREC-Coling 2024

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04032v2) [paper-pdf](http://arxiv.org/pdf/2405.04032v2)

**Authors**: Chunyan Zheng, Keke Sun, Wenhao Zhao, Haibo Zhou, Lixin Jiang, Shaoyang Song, Chunlai Zhou

**Abstract**: Large pretrained language models (LLMs) have shown surprising In-Context Learning (ICL) ability. An important application in deploying large language models is to augment LLMs with a private database for some specific task. The main problem with this promising commercial use is that LLMs have been shown to memorize their training data and their prompt data are vulnerable to membership inference attacks (MIA) and prompt leaking attacks. In order to deal with this problem, we treat LLMs as untrusted in privacy and propose a locally differentially private framework of in-context learning(LDP-ICL) in the settings where labels are sensitive. Considering the mechanisms of in-context learning in Transformers by gradient descent, we provide an analysis of the trade-off between privacy and utility in such LDP-ICL for classification. Moreover, we apply LDP-ICL to the discrete distribution estimation problem. In the end, we perform several experiments to demonstrate our analysis results.



## **2. Air Gap: Protecting Privacy-Conscious Conversational Agents**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.



## **3. Critical Infrastructure Protection: Generative AI, Challenges, and Opportunities**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04874v1) [paper-pdf](http://arxiv.org/pdf/2405.04874v1)

**Authors**: Yagmur Yigit, Mohamed Amine Ferrag, Iqbal H. Sarker, Leandros A. Maglaras, Christos Chrysoulas, Naghmeh Moradpoor, Helge Janicke

**Abstract**: Critical National Infrastructure (CNI) encompasses a nation's essential assets that are fundamental to the operation of society and the economy, ensuring the provision of vital utilities such as energy, water, transportation, and communication. Nevertheless, growing cybersecurity threats targeting these infrastructures can potentially interfere with operations and seriously risk national security and public safety. In this paper, we examine the intricate issues raised by cybersecurity risks to vital infrastructure, highlighting these systems' vulnerability to different types of cyberattacks. We analyse the significance of trust, privacy, and resilience for Critical Infrastructure Protection (CIP), examining the diverse standards and regulations to manage these domains. We also scrutinise the co-analysis of safety and security, offering innovative approaches for their integration and emphasising the interdependence between these fields. Furthermore, we introduce a comprehensive method for CIP leveraging Generative AI and Large Language Models (LLMs), giving a tailored lifecycle and discussing specific applications across different critical infrastructure sectors. Lastly, we discuss potential future directions that promise to enhance the security and resilience of critical infrastructures. This paper proposes innovative strategies for CIP from evolving attacks and enhances comprehension of cybersecurity concerns related to critical infrastructure.



## **4. Large Language Models for Cyber Security: A Systematic Literature Review**

cs.CR

46 pages,6 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04760v1) [paper-pdf](http://arxiv.org/pdf/2405.04760v1)

**Authors**: HanXiang Xu, ShenAo Wang, Ningke Li, Yanjie Zhao, Kai Chen, Kailong Wang, Yang Liu, Ting Yu, HaoYu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.



## **5. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.



## **6. AttacKG+:Boosting Attack Knowledge Graph Construction with Large Language Models**

cs.CR

20 pages, 5 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04753v1) [paper-pdf](http://arxiv.org/pdf/2405.04753v1)

**Authors**: Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang Wang, Yi Xie, Guozheng Yang, Yuliang Lu, Ee-Chien Chang

**Abstract**: Attack knowledge graph construction seeks to convert textual cyber threat intelligence (CTI) reports into structured representations, portraying the evolutionary traces of cyber attacks. Even though previous research has proposed various methods to construct attack knowledge graphs, they generally suffer from limited generalization capability to diverse knowledge types as well as requirement of expertise in model design and tuning. Addressing these limitations, we seek to utilize Large Language Models (LLMs), which have achieved enormous success in a broad range of tasks given exceptional capabilities in both language understanding and zero-shot task fulfillment. Thus, we propose a fully automatic LLM-based framework to construct attack knowledge graphs named: AttacKG+. Our framework consists of four consecutive modules: rewriter, parser, identifier, and summarizer, each of which is implemented by instruction prompting and in-context learning empowered by LLMs. Furthermore, we upgrade the existing attack knowledge schema and propose a comprehensive version. We represent a cyber attack as a temporally unfolding event, each temporal step of which encapsulates three layers of representation, including behavior graph, MITRE TTP labels, and state summary. Extensive evaluation demonstrates that: 1) our formulation seamlessly satisfies the information needs in threat event analysis, 2) our construction framework is effective in faithfully and accurately extracting the information defined by AttacKG+, and 3) our attack graph directly benefits downstream security practices such as attack reconstruction. All the code and datasets will be released upon acceptance.



## **7. Revisiting character-level adversarial attacks**

cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.



## **8. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.



## **9. Are aligned neural networks adversarially aligned?**

cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.



## **10. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.



## **11. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.



## **12. Confidential and Protected Disease Classifier using Fully Homomorphic Encryption**

cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02790v1) [paper-pdf](http://arxiv.org/pdf/2405.02790v1)

**Authors**: Aditya Malik, Nalini Ratha, Bharat Yalavarthi, Tilak Sharma, Arjun Kaushik, Charanjit Jutla

**Abstract**: With the rapid surge in the prevalence of Large Language Models (LLMs), individuals are increasingly turning to conversational AI for initial insights across various domains, including health-related inquiries such as disease diagnosis. Many users seek potential causes on platforms like ChatGPT or Bard before consulting a medical professional for their ailment. These platforms offer valuable benefits by streamlining the diagnosis process, alleviating the significant workload of healthcare practitioners, and saving users both time and money by avoiding unnecessary doctor visits. However, Despite the convenience of such platforms, sharing personal medical data online poses risks, including the presence of malicious platforms or potential eavesdropping by attackers. To address privacy concerns, we propose a novel framework combining FHE and Deep Learning for a secure and private diagnosis system. Operating on a question-and-answer-based model akin to an interaction with a medical practitioner, this end-to-end secure system employs Fully Homomorphic Encryption (FHE) to handle encrypted input data. Given FHE's computational constraints, we adapt deep neural networks and activation functions to the encryted domain. Further, we also propose a faster algorithm to compute summation of ciphertext elements. Through rigorous experiments, we demonstrate the efficacy of our approach. The proposed framework achieves strict security and privacy with minimal loss in performance.



## **13. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.



## **14. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

cs.LG

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2310.06387v2) [paper-pdf](http://arxiv.org/pdf/2310.06387v2)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating harmful content have emerged. In this paper, we delve into the potential of In-Context Learning (ICL) to modulate the alignment of LLMs. Specifically, we propose the In-Context Attack (ICA), which employs strategically crafted harmful demonstrations to subvert LLMs, and the In-Context Defense (ICD), which bolsters model resilience through examples that demonstrate refusal to produce harmful responses. Through extensive experiments, we demonstrate the efficacy of ICA and ICD in respectively elevating and mitigating the success rates of jailbreaking prompts. Moreover, we offer theoretical insights into the mechanism by which a limited set of in-context demonstrations can pivotally influence the safety alignment of LLMs. Our findings illuminate the profound influence of ICL on LLM behavior, opening new avenues for improving the safety and alignment of LLMs.



## **15. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

cs.SE

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02580v1) [paper-pdf](http://arxiv.org/pdf/2405.02580v1)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs, such as GPT-4, to transfer existing human-written properties (e.g., those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new prop- erty for a given code. While this basic process is relatively straight- forward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) runtime-verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated prop- erties. We have implemented these strategies into a novel system called PropertyGPT, with 623 human-written properties collected from 23 Certora projects. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, resulting in $8,256 bug bounty rewards.



## **16. Adaptive and robust watermark against model extraction attack**

cs.CR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.02365v1) [paper-pdf](http://arxiv.org/pdf/2405.02365v1)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai

**Abstract**: Large language models have boosted Large Models as a Service (LMaaS) into a thriving business sector. But even model owners offering only API access while keeping model parameters and internal workings private, their Intellectual Property (IP) are still at risk of theft through model extraction attacks. To safeguard the IP of these models and mitigate unfair competition in the language model market, watermarking technology serves as an efficient post-hoc solution for identifying IP infringements. However, existing IP protection watermarking methods either explicitly alter the original output of the language model or implant watermark signals in the model logits. These methods forcefully distort the original distribution of the language model and impact the sampling process, leading to a decline in the quality of the generated text. The existing method also fails to achieve end-to-end adaptive watermark embedding and lack robustness verification in complex scenarios where watermark detection is subject to interference. To overcome these challenges, we propose PromptShield, a plug-and-play IP protection watermarking method to resist model extraction attacks without training additional modules. Leveraging the self-reminding properties inherent in large language models, we encapsulate the user's query with a watermark self-generated instruction, nudging the LLMs to automatically generate watermark words in its output without compromising generation quality. Our method does not require access to the model's internal logits and minimizes alterations to the model's distribution using prompt-guided cues. Comprehensive experimental results consistently demonstrate the effectiveness, harmlessness, and robustness of our watermark. Moreover, Our watermark detection method remains robust and high detection sensitivity even when subjected to interference.



## **17. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2312.03853v3) [paper-pdf](http://arxiv.org/pdf/2312.03853v3)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then followed a role-play style to elicit prohibited responses. By making use of personas, we show that such responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **18. Generative AI in Cybersecurity**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01674v1) [paper-pdf](http://arxiv.org/pdf/2405.01674v1)

**Authors**: Shivani Metta, Isaac Chang, Jack Parker, Michael P. Roman, Arturo F. Ehuan

**Abstract**: The dawn of Generative Artificial Intelligence (GAI), characterized by advanced models such as Generative Pre-trained Transformers (GPT) and other Large Language Models (LLMs), has been pivotal in reshaping the field of data analysis, pattern recognition, and decision-making processes. This surge in GAI technology has ushered in not only innovative opportunities for data processing and automation but has also introduced significant cybersecurity challenges.   As GAI rapidly progresses, it outstrips the current pace of cybersecurity protocols and regulatory frameworks, leading to a paradox wherein the same innovations meant to safeguard digital infrastructures also enhance the arsenal available to cyber criminals. These adversaries, adept at swiftly integrating and exploiting emerging technologies, may utilize GAI to develop malware that is both more covert and adaptable, thus complicating traditional cybersecurity efforts.   The acceleration of GAI presents an ambiguous frontier for cybersecurity experts, offering potent tools for threat detection and response, while concurrently providing cyber attackers with the means to engineer more intricate and potent malware. Through the joint efforts of Duke Pratt School of Engineering, Coalfire, and Safebreach, this research undertakes a meticulous analysis of how malicious agents are exploiting GAI to augment their attack strategies, emphasizing a critical issue for the integrity of future cybersecurity initiatives. The study highlights the critical need for organizations to proactively identify and develop more complex defensive strategies to counter the sophisticated employment of GAI in malware creation.



## **19. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense



## **20. Boosting Jailbreak Attack with Momentum**

cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.



## **21. Adversarial Attacks and Defense for Conversation Entailment Task**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.



## **22. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2404.07921v2) [paper-pdf](http://arxiv.org/pdf/2404.07921v2)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **23. Assessing LLMs in Malicious Code Deobfuscation of Real-world Malware Campaigns**

cs.CR

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19715v1) [paper-pdf](http://arxiv.org/pdf/2404.19715v1)

**Authors**: Constantinos Patsakis, Fran Casino, Nikolaos Lykousas

**Abstract**: The integration of large language models (LLMs) into various pipelines is increasingly widespread, effectively automating many manual tasks and often surpassing human capabilities. Cybersecurity researchers and practitioners have recognised this potential. Thus, they are actively exploring its applications, given the vast volume of heterogeneous data that requires processing to identify anomalies, potential bypasses, attacks, and fraudulent incidents. On top of this, LLMs' advanced capabilities in generating functional code, comprehending code context, and summarising its operations can also be leveraged for reverse engineering and malware deobfuscation. To this end, we delve into the deobfuscation capabilities of state-of-the-art LLMs. Beyond merely discussing a hypothetical scenario, we evaluate four LLMs with real-world malicious scripts used in the notorious Emotet malware campaign. Our results indicate that while not absolutely accurate yet, some LLMs can efficiently deobfuscate such payloads. Thus, fine-tuning LLMs for this task can be a viable potential for future AI-powered threat intelligence pipelines in the fight against obfuscated malware.



## **24. Transferring Troubles: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

cs.CL

work in progress

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19597v1) [paper-pdf](http://arxiv.org/pdf/2404.19597v1)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. However, the impact of backdoor attacks on multilingual models remains under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data in one or two languages can affect the outputs in languages whose instruction-tuning data was not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5, BLOOM, and GPT-3.5-turbo, with high attack success rates, surpassing 95% in several languages across various scenarios. Alarmingly, our findings also indicate that larger models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments show that triggers can still work even after paraphrasing, and the backdoor mechanism proves highly effective in cross-lingual response settings across 25 languages, achieving an average attack success rate of 50%. Our study aims to highlight the vulnerabilities and significant security risks present in current multilingual LLMs, underscoring the emergent need for targeted security measures.



## **25. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-04-30    [abs](http://arxiv.org/abs/2404.19287v1) [paper-pdf](http://arxiv.org/pdf/2404.19287v1)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.



## **26. Intention Analysis Makes LLMs A Good Jailbreak Defender**

cs.CL

20 pages, 16 figures

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2401.06561v3) [paper-pdf](http://arxiv.org/pdf/2401.06561v3)

**Authors**: Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao

**Abstract**: Aligning large language models (LLMs) with human values, particularly in the face of complex and stealthy jailbreak attacks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis ($\mathbb{IA}$). The principle behind this is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, $\mathbb{IA}$ is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on varying jailbreak benchmarks across ChatGLM, LLaMA2, Vicuna, MPT, DeepSeek, and GPT-3.5 show that $\mathbb{IA}$ could consistently and significantly reduce the harmfulness in responses (averagely -53.1% attack success rate) and maintain the general helpfulness. Encouragingly, with the help of our $\mathbb{IA}$, Vicuna-7B even outperforms GPT-3.5 in terms of attack success rate. Further analyses present some insights into how our method works. To facilitate reproducibility, we release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis.



## **27. AppPoet: Large Language Model based Android malware detection via multi-view prompt engineering**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18816v1) [paper-pdf](http://arxiv.org/pdf/2404.18816v1)

**Authors**: Wenxiang Zhao, Juntao Wu, Zhaoyi Meng

**Abstract**: Due to the vast array of Android applications, their multifarious functions and intricate behavioral semantics, attackers can adopt various tactics to conceal their genuine attack intentions within legitimate functions. However, numerous feature engineering based methods suffer from a limitation in mining behavioral semantic information, thus impeding the accuracy and efficiency of Android malware detection. Besides, the majority of existing feature engineering based methods are weakly interpretive and fail to furnish researchers with effective and readable detection reports. Inspired by the success of the Large Language Models (LLMs) in natural language understanding, we propose AppPoet, a LLM-assisted multi-view system for Android malware detection. Firstly, AppPoet employs a static method to comprehensively collect application features and formulate various observation views. Subsequently, it steers the LLM to produce function descriptions and behavioral summaries for views via our meticulously devised multi-view prompt engineering technique to realize the deep mining of view semantics. Finally, we collaboratively fuse the multi-view information to efficiently and accurately detect malware through a deep neural network (DNN) classifier and then generate the heuristic diagnostic reports. Experimental results demonstrate that our method achieves a detection accuracy of 97.15% and an F1 score of 97.21%, which is superior to the baseline method Drebin and its variant. Furthermore, the case study evaluates the effectiveness of our generated diagnostic reports.



## **28. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2311.14455v4) [paper-pdf](http://arxiv.org/pdf/2311.14455v4)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **29. Assessing Cybersecurity Vulnerabilities in Code Large Language Models**

cs.CR

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2404.18567v1) [paper-pdf](http://arxiv.org/pdf/2404.18567v1)

**Authors**: Md Imran Hossen, Jianyi Zhang, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Code Large Language Models (Code LLMs) are increasingly utilized as AI coding assistants and integrated into various applications. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. To bridge this gap, this paper presents EvilInstructCoder, a framework specifically designed to assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs to adversarial attacks. EvilInstructCoder introduces the Adversarial Code Injection Engine to automatically generate malicious code snippets and inject them into benign code to poison instruction tuning datasets. It incorporates practical threat models to reflect real-world adversaries with varying capabilities and evaluates the exploitability of instruction-tuned Code LLMs under these diverse adversarial attack scenarios. Through the use of EvilInstructCoder, we conduct a comprehensive investigation into the exploitability of instruction tuning for coding tasks using three state-of-the-art Code LLM models: CodeLlama, DeepSeek-Coder, and StarCoder2, under various adversarial attack scenarios. Our experimental results reveal a significant vulnerability in these models, demonstrating that adversaries can manipulate the models to generate malicious payloads within benign code contexts in response to natural language instructions. For instance, under the backdoor attack setting, by poisoning only 81 samples (0.5\% of the entire instruction dataset), we achieve Attack Success Rate at 1 (ASR@1) scores ranging from 76\% to 86\% for different model families. Our study sheds light on the critical cybersecurity vulnerabilities posed by instruction-tuned Code LLMs and emphasizes the urgent necessity for robust defense mechanisms to mitigate the identified vulnerabilities.



## **30. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

cs.LG

**SubmitDate**: 2024-04-29    [abs](http://arxiv.org/abs/2401.10862v2) [paper-pdf](http://arxiv.org/pdf/2401.10862v2)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: Large Language Models (LLMs) are susceptible to `jailbreaking' prompts, which can induce the generation of harmful content. This paper demonstrates that moderate WANDA pruning (Sun et al., 2023) can increase their resistance to such attacks without the need for fine-tuning, while maintaining performance on standard benchmarks. Our findings suggest that the benefits of pruning correlate with the initial safety levels of the model, indicating a regularizing effect of WANDA pruning. We introduce a dataset of 225 harmful tasks across five categories to systematically evaluate this safety enhancement. We argue that safety improvements can be understood through a regularization perspective. First, we show that pruning helps LLMs focus more effectively on task-relevant tokens within jailbreaking prompts. Then, we analyze the effects of pruning on the perplexity of malicious prompts before and after their integration into jailbreak templates. Finally, we demonstrate statistically significant performance improvements under domain shifts when applying WANDA to linear models.



## **31. Learnable Linguistic Watermarks for Tracing Model Extraction Attacks on Large Language Models**

cs.CR

not decided

**SubmitDate**: 2024-04-28    [abs](http://arxiv.org/abs/2405.01509v1) [paper-pdf](http://arxiv.org/pdf/2405.01509v1)

**Authors**: Minhao Bai, Kaiyi Pang, Yongfeng Huang

**Abstract**: In the rapidly evolving domain of artificial intelligence, safeguarding the intellectual property of Large Language Models (LLMs) is increasingly crucial. Current watermarking techniques against model extraction attacks, which rely on signal insertion in model logits or post-processing of generated text, remain largely heuristic. We propose a novel method for embedding learnable linguistic watermarks in LLMs, aimed at tracing and preventing model extraction attacks. Our approach subtly modifies the LLM's output distribution by introducing controlled noise into token frequency distributions, embedding an statistically identifiable controllable watermark.We leverage statistical hypothesis testing and information theory, particularly focusing on Kullback-Leibler Divergence, to differentiate between original and modified distributions effectively. Our watermarking method strikes a delicate well balance between robustness and output quality, maintaining low false positive/negative rates and preserving the LLM's original performance.



## **32. Investigating the prompt leakage effect and black-box defenses for multi-turn LLM interactions**

cs.CR

**SubmitDate**: 2024-04-26    [abs](http://arxiv.org/abs/2404.16251v2) [paper-pdf](http://arxiv.org/pdf/2404.16251v2)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Philippe Laban, Ben Risher, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu

**Abstract**: Prompt leakage in large language models (LLMs) poses a significant security and privacy threat, particularly in retrieval-augmented generation (RAG) systems. However, leakage in multi-turn LLM interactions along with mitigation strategies has not been studied in a standardized manner. This paper investigates LLM vulnerabilities against prompt leakage across 4 diverse domains and 10 closed- and open-source LLMs. Our unique multi-turn threat model leverages the LLM's sycophancy effect and our analysis dissects task instruction and knowledge leakage in the LLM response. In a multi-turn setting, our threat model elevates the average attack success rate (ASR) to 86.2%, including a 99% leakage with GPT-4 and claude-1.3. We find that some black-box LLMs like Gemini show variable susceptibility to leakage across domains - they are more likely to leak contextual knowledge in the news domain compared to the medical domain. Our experiments measure specific effects of 6 black-box defense strategies, including a query-rewriter in the RAG scenario. Our proposed multi-tier combination of defenses still has an ASR of 5.3% for black-box LLMs, indicating room for enhancement and future direction for LLM security research.



## **33. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

cs.CL

**SubmitDate**: 2024-04-25    [abs](http://arxiv.org/abs/2404.16369v1) [paper-pdf](http://arxiv.org/pdf/2404.16369v1)

**Authors**: Yukai Zhou, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is crucial to generating responses consistent with human values. Despite their ability to recognize and avoid harmful queries, LLMs are vulnerable to "jailbreaking" attacks, where carefully crafted prompts elicit them to produce toxic content. One category of jailbreak attacks is reformulating the task as adversarial attacks by eliciting the LLM to generate an affirmative response. However, the typical attack in this category GCG has very limited attack success rate. In this study, to better study the jailbreak attack, we introduce the DSN (Don't Say No) attack, which prompts LLMs to not only generate affirmative responses but also novelly enhance the objective to suppress refusals. In addition, another challenge lies in jailbreak attacks is the evaluation, as it is difficult to directly and accurately assess the harmfulness of the attack. The existing evaluation such as refusal keyword matching has its own limitation as it reveals numerous false positive and false negative instances. To overcome this challenge, we propose an ensemble evaluation pipeline incorporating Natural Language Inference (NLI) contradiction assessment and two external LLM evaluators. Extensive experiments demonstrate the potency of the DSN and the effectiveness of ensemble evaluation compared to baseline methods.



## **34. Attacks on Third-Party APIs of Large Language Models**

cs.CR

ICLR 2024 Workshop on Secure and Trustworthy Large Language Models

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.16891v1) [paper-pdf](http://arxiv.org/pdf/2404.16891v1)

**Authors**: Wanru Zhao, Vidit Khazanchi, Haodi Xing, Xuanli He, Qiongkai Xu, Nicholas Donald Lane

**Abstract**: Large language model (LLM) services have recently begun offering a plugin ecosystem to interact with third-party API services. This innovation enhances the capabilities of LLMs, but it also introduces risks, as these plugins developed by various third parties cannot be easily trusted. This paper proposes a new attacking framework to examine security and safety vulnerabilities within LLM platforms that incorporate third-party services. Applying our framework specifically to widely used LLMs, we identify real-world malicious attacks across various domains on third-party APIs that can imperceptibly modify LLM outputs. The paper discusses the unique challenges posed by third-party API integration and offers strategic possibilities to improve the security and safety of LLM ecosystems moving forward. Our code is released at https://github.com/vk0812/Third-Party-Attacks-on-LLMs.



## **35. Talk Too Much: Poisoning Large Language Models under Token Limit**

cs.CL

**SubmitDate**: 2024-04-24    [abs](http://arxiv.org/abs/2404.14795v2) [paper-pdf](http://arxiv.org/pdf/2404.14795v2)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.



## **36. Large Language Models Spot Phishing Emails with Surprising Accuracy: A Comparative Analysis of Performance**

cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15485v1) [paper-pdf](http://arxiv.org/pdf/2404.15485v1)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.



## **37. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.01318v2) [paper-pdf](http://arxiv.org/pdf/2404.01318v2)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.



## **38. Versatile Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

cs.CV

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2306.00816v3) [paper-pdf](http://arxiv.org/pdf/2306.00816v3)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed \textit{backdoor attack}. Currently, implementing backdoor attacks in physical scenarios still faces significant challenges. Physical attacks are labor-intensive and time-consuming, and the triggers are selected in a manual and heuristic way. Moreover, expanding digital attacks to physical scenarios faces many challenges due to their sensitivity to visual distortions and the absence of counterparts in the real world. To address these challenges, we define a novel trigger called the \textbf{V}isible, \textbf{S}emantic, \textbf{S}ample-Specific, and \textbf{C}ompatible (VSSC) trigger, to achieve effective, stealthy and robust simultaneously, which can also be effectively deployed in the physical scenario using corresponding objects. To implement the VSSC trigger, we propose an automated pipeline comprising three modules: a trigger selection module that systematically identifies suitable triggers leveraging large language models, a trigger insertion module that employs generative models to seamlessly integrate triggers into images, and a quality assessment module that ensures the natural and successful insertion of triggers through vision-language models. Extensive experimental results and analysis validate the effectiveness, stealthiness, and robustness of the VSSC trigger. It can not only maintain robustness under visual distortions but also demonstrates strong practicality in the physical scenario. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.



## **39. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14304v1) [paper-pdf](http://arxiv.org/pdf/2404.14304v1)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.



## **40. Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models**

cs.CR

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.12916v2) [paper-pdf](http://arxiv.org/pdf/2404.12916v2)

**Authors**: Zhenyang Ni, Rui Ye, Yuxi Wei, Zhen Xiang, Yanfeng Wang, Siheng Chen

**Abstract**: Vision-Large-Language-models(VLMs) have great application prospects in autonomous driving. Despite the ability of VLMs to comprehend and make decisions in complex scenarios, their integration into safety-critical autonomous driving systems poses serious security risks. In this paper, we propose BadVLMDriver, the first backdoor attack against VLMs for autonomous driving that can be launched in practice using physical objects. Unlike existing backdoor attacks against VLMs that rely on digital modifications, BadVLMDriver uses common physical items, such as a red balloon, to induce unsafe actions like sudden acceleration, highlighting a significant real-world threat to autonomous vehicle safety. To execute BadVLMDriver, we develop an automated pipeline utilizing natural language instructions to generate backdoor training samples with embedded malicious behaviors. This approach allows for flexible trigger and behavior selection, enhancing the stealth and practicality of the attack in diverse scenarios. We conduct extensive experiments to evaluate BadVLMDriver for two representative VLMs, five different trigger objects, and two types of malicious backdoor behaviors. BadVLMDriver achieves a 92% attack success rate in inducing a sudden acceleration when coming across a pedestrian holding a red balloon. Thus, BadVLMDriver not only demonstrates a critical security risk but also emphasizes the urgent need for developing robust defense mechanisms to protect against such vulnerabilities in autonomous driving technologies.



## **41. Protecting Your LLMs with Information Bottleneck**

cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13968v1) [paper-pdf](http://arxiv.org/pdf/2404.13968v1)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **42. Typos that Broke the RAG's Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations**

cs.CL

Under Review

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13948v1) [paper-pdf](http://arxiv.org/pdf/2404.13948v1)

**Authors**: Sukmin Cho, Soyeong Jeong, Jeongyeon Seo, Taeho Hwang, Jong C. Park

**Abstract**: The robustness of recent Large Language Models (LLMs) has become increasingly crucial as their applicability expands across various domains and real-world applications. Retrieval-Augmented Generation (RAG) is a promising solution for addressing the limitations of LLMs, yet existing studies on the robustness of RAG often overlook the interconnected relationships between RAG components or the potential threats prevalent in real-world databases, such as minor textual errors. In this work, we investigate two underexplored aspects when assessing the robustness of RAG: 1) vulnerability to noisy documents through low-level perturbations and 2) a holistic evaluation of RAG robustness. Furthermore, we introduce a novel attack method, the Genetic Attack on RAG (\textit{GARAG}), which targets these aspects. Specifically, GARAG is designed to reveal vulnerabilities within each component and test the overall system functionality against noisy documents. We validate RAG robustness by applying our \textit{GARAG} to standard QA datasets, incorporating diverse retrievers and LLMs. The experimental results show that GARAG consistently achieves high attack success rates. Also, it significantly devastates the performance of each component and their synergy, highlighting the substantial risk that minor textual inaccuracies pose in disrupting RAG systems in the real world.



## **43. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

cs.CL

Competition Report

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14461v1) [paper-pdf](http://arxiv.org/pdf/2404.14461v1)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.



## **44. Bot or Human? Detecting ChatGPT Imposters with A Single Question**

cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2305.06424v3) [paper-pdf](http://arxiv.org/pdf/2305.06424v3)

**Authors**: Hong Wang, Xuan Luo, Weizhi Wang, Xifeng Yan

**Abstract**: Large language models like GPT-4 have recently demonstrated impressive capabilities in natural language understanding and generation, enabling various applications including translation, essay writing, and chit-chatting. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large Language Model Authenticity via a Single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, and ASCII art reasoning), and those that are easy for bots but difficult for humans (e.g., memorization and computation). Our approach shows different strengths of these questions in their effectiveness, providing a new way for online service providers to protect themselves against nefarious activities and ensure that they are serving real users. We open-sourced our code and dataset on https://github.com/hongwang600/FLAIR and welcome contributions from the community.



## **45. AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs**

cs.CR

32 pages, 9 figures, 7 tables

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.16873v1) [paper-pdf](http://arxiv.org/pdf/2404.16873v1)

**Authors**: Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, Yuandong Tian

**Abstract**: While recently Large Language Models (LLMs) have achieved remarkable successes, they are vulnerable to certain jailbreaking attacks that lead to generation of inappropriate or harmful content. Manual red-teaming requires finding adversarial prompts that cause such jailbreaking, e.g. by appending a suffix to a given instruction, which is inefficient and time-consuming. On the other hand, automatic adversarial prompt generation often leads to semantically meaningless attacks that can easily be detected by perplexity-based filters, may require gradient information from the TargetLLM, or do not scale well due to time-consuming discrete optimization processes over the token space. In this paper, we present a novel method that uses another LLM, called the AdvPrompter, to generate human-readable adversarial prompts in seconds, $\sim800\times$ faster than existing optimization-based approaches. We train the AdvPrompter using a novel algorithm that does not require access to the gradients of the TargetLLM. This process alternates between two steps: (1) generating high-quality target adversarial suffixes by optimizing the AdvPrompter predictions, and (2) low-rank fine-tuning of the AdvPrompter with the generated adversarial suffixes. The trained AdvPrompter generates suffixes that veil the input instruction without changing its meaning, such that the TargetLLM is lured to give a harmful response. Experimental results on popular open source TargetLLMs show state-of-the-art results on the AdvBench dataset, that also transfer to closed-source black-box LLM APIs. Further, we demonstrate that by fine-tuning on a synthetic dataset generated by AdvPrompter, LLMs can be made more robust against jailbreaking attacks while maintaining performance, i.e. high MMLU scores.



## **46. Trojan Detection in Large Language Models: Insights from The Trojan Detection Challenge**

cs.CL

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13660v1) [paper-pdf](http://arxiv.org/pdf/2404.13660v1)

**Authors**: Narek Maloyan, Ekansh Verma, Bulat Nutfullin, Bislan Ashinov

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various domains, but their vulnerability to trojan or backdoor attacks poses significant security risks. This paper explores the challenges and insights gained from the Trojan Detection Competition 2023 (TDC2023), which focused on identifying and evaluating trojan attacks on LLMs. We investigate the difficulty of distinguishing between intended and unintended triggers, as well as the feasibility of reverse engineering trojans in real-world scenarios. Our comparative analysis of various trojan detection methods reveals that achieving high Recall scores is significantly more challenging than obtaining high Reverse-Engineering Attack Success Rate (REASR) scores. The top-performing methods in the competition achieved Recall scores around 0.16, comparable to a simple baseline of randomly sampling sentences from a distribution similar to the given training prefixes. This finding raises questions about the detectability and recoverability of trojans inserted into the model, given only the harmful targets. Despite the inability to fully solve the problem, the competition has led to interesting observations about the viability of trojan detection and improved techniques for optimizing LLM input prompts. The phenomenon of unintended triggers and the difficulty in distinguishing them from intended triggers highlights the need for further research into the robustness and interpretability of LLMs. The TDC2023 has provided valuable insights into the challenges and opportunities associated with trojan detection in LLMs, laying the groundwork for future research in this area to ensure their safety and reliability in real-world applications.



## **47. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2403.14280v3) [paper-pdf](http://arxiv.org/pdf/2403.14280v3)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **48. Intrusion Detection at Scale with the Assistance of a Command-line Language Model**

cs.CR

Accepted by IEEE/IFIP International Conference on Dependable Systems  and Networks (DSN), industry track

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13402v1) [paper-pdf](http://arxiv.org/pdf/2404.13402v1)

**Authors**: Jiongliang Lin, Yiwen Guo, Hao Chen

**Abstract**: Intrusion detection is a long standing and crucial problem in security. A system capable of detecting intrusions automatically is on great demand in enterprise security solutions. Existing solutions rely heavily on hand-crafted rules designed by security operators, which suffer from high false negative rates and poor generalization ability to new, zero-day attacks at scale. AI and machine learning offer promising solutions to address the issues, by inspecting abnormal user behaviors intelligently and automatically from data. However, existing learning-based intrusion detection systems in the literature are mostly designed for small data, and they lack the ability to leverage the power of big data in cloud environments. In this paper, we target at this problem and introduce an intrusion detection system which incorporates large-scale pre-training, so as to train a large language model based on tens of millions of command lines for AI-based intrusion detection. Experiments performed on 30 million training samples and 10 million test samples verify the effectiveness of our solution.



## **49. ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs**

cs.CL

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2402.11753v3) [paper-pdf](http://arxiv.org/pdf/2402.11753v3)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Zhen Xiang, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**Abstract**: Safety is critical to the usage of large language models (LLMs). Multiple techniques such as data filtering and supervised fine-tuning have been developed to strengthen LLM safety. However, currently known techniques presume that corpora used for safety alignment of LLMs are solely interpreted by semantics. This assumption, however, does not hold in real-world applications, which leads to severe vulnerabilities in LLMs. For example, users of forums often use ASCII art, a form of text-based art, to convey image information. In this paper, we propose a novel ASCII art-based jailbreak attack and introduce a comprehensive benchmark Vision-in-Text Challenge (ViTC) to evaluate the capabilities of LLMs in recognizing prompts that cannot be solely interpreted by semantics. We show that five SOTA LLMs (GPT-3.5, GPT-4, Gemini, Claude, and Llama2) struggle to recognize prompts provided in the form of ASCII art. Based on this observation, we develop the jailbreak attack ArtPrompt, which leverages the poor performance of LLMs in recognizing ASCII art to bypass safety measures and elicit undesired behaviors from LLMs. ArtPrompt only requires black-box access to the victim LLMs, making it a practical attack. We evaluate ArtPrompt on five SOTA LLMs, and show that ArtPrompt can effectively and efficiently induce undesired behaviors from all five LLMs. Our code is available at https://github.com/uw-nsl/ArtPrompt.



## **50. CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models**

cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.13161v1) [paper-pdf](http://arxiv.org/pdf/2404.13161v1)

**Authors**: Manish Bhatt, Sahana Chennabasappa, Yue Li, Cyrus Nikolaidis, Daniel Song, Shengye Wan, Faizan Ahmad, Cornelius Aschermann, Yaohui Chen, Dhaval Kapil, David Molnar, Spencer Whitman, Joshua Saxe

**Abstract**: Large language models (LLMs) introduce new security risks, but there are few comprehensive evaluation suites to measure and reduce these risks. We present BenchmarkName, a novel benchmark to quantify LLM security risks and capabilities. We introduce two new areas for testing: prompt injection and code interpreter abuse. We evaluated multiple state-of-the-art (SOTA) LLMs, including GPT-4, Mistral, Meta Llama 3 70B-Instruct, and Code Llama. Our results show that conditioning away risk of attack remains an unsolved problem; for example, all tested models showed between 26% and 41% successful prompt injection tests. We further introduce the safety-utility tradeoff: conditioning an LLM to reject unsafe prompts can cause the LLM to falsely reject answering benign prompts, which lowers utility. We propose quantifying this tradeoff using False Refusal Rate (FRR). As an illustration, we introduce a novel test set to quantify FRR for cyberattack helpfulness risk. We find many LLMs able to successfully comply with "borderline" benign requests while still rejecting most unsafe requests. Finally, we quantify the utility of LLMs for automating a core cybersecurity task, that of exploiting software vulnerabilities. This is important because the offensive capabilities of LLMs are of intense interest; we quantify this by creating novel test sets for four representative problems. We find that models with coding capabilities perform better than those without, but that further work is needed for LLMs to become proficient at exploit generation. Our code is open source and can be used to evaluate other LLMs.



