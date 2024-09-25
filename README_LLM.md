# Latest Large Language Model Attack Papers
**update at 2024-09-25 10:22:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Order of Magnitude Speedups for LLM Membership Inference**

cs.LG

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14513v2) [paper-pdf](http://arxiv.org/pdf/2409.14513v2)

**Authors**: Rongting Zhang, Martin Bertran, Aaron Roth

**Abstract**: Large Language Models (LLMs) have the promise to revolutionize computing broadly, but their complexity and extensive training data also expose significant privacy vulnerabilities. One of the simplest privacy risks associated with LLMs is their susceptibility to membership inference attacks (MIAs), wherein an adversary aims to determine whether a specific data point was part of the model's training set. Although this is a known risk, state of the art methodologies for MIAs rely on training multiple computationally costly shadow models, making risk evaluation prohibitive for large models. Here we adapt a recent line of work which uses quantile regression to mount membership inference attacks; we extend this work by proposing a low-cost MIA that leverages an ensemble of small quantile regression models to determine if a document belongs to the model's training set or not. We demonstrate the effectiveness of this approach on fine-tuned LLMs of varying families (OPT, Pythia, Llama) and across multiple datasets. Across all scenarios we obtain comparable or improved accuracy compared to state of the art shadow model approaches, with as little as 6% of their computation budget. We demonstrate increased effectiveness across multi-epoch trained target models, and architecture miss-specification robustness, that is, we can mount an effective attack against a model using a different tokenizer and architecture, without requiring knowledge on the target model.



## **2. Cyber Knowledge Completion Using Large Language Models**

cs.CR

7 pages, 2 figures. Submitted to 2024 IEEE International Conference  on Big Data

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16176v1) [paper-pdf](http://arxiv.org/pdf/2409.16176v1)

**Authors**: Braden K Webb, Sumit Purohit, Rounak Meyur

**Abstract**: The integration of the Internet of Things (IoT) into Cyber-Physical Systems (CPSs) has expanded their cyber-attack surface, introducing new and sophisticated threats with potential to exploit emerging vulnerabilities. Assessing the risks of CPSs is increasingly difficult due to incomplete and outdated cybersecurity knowledge. This highlights the urgent need for better-informed risk assessments and mitigation strategies. While previous efforts have relied on rule-based natural language processing (NLP) tools to map vulnerabilities, weaknesses, and attack patterns, recent advancements in Large Language Models (LLMs) present a unique opportunity to enhance cyber-attack knowledge completion through improved reasoning, inference, and summarization capabilities. We apply embedding models to encapsulate information on attack patterns and adversarial techniques, generating mappings between them using vector embeddings. Additionally, we propose a Retrieval-Augmented Generation (RAG)-based approach that leverages pre-trained models to create structured mappings between different taxonomies of threat patterns. Further, we use a small hand-labeled dataset to compare the proposed RAG-based approach to a baseline standard binary classification model. Thus, the proposed approach provides a comprehensive framework to address the challenge of cyber-attack knowledge graph completion.



## **3. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack**

cs.CR

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2404.01833v2) [paper-pdf](http://arxiv.org/pdf/2404.01833v2)

**Authors**: Mark Russinovich, Ahmed Salem, Ronen Eldan

**Abstract**: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.



## **4. Large Language Models as Carriers of Hidden Messages**

cs.CL

Work in progress. Code is available at  https://github.com/j-hoscilowic/zurek-stegano

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2406.02481v4) [paper-pdf](http://arxiv.org/pdf/2406.02481v4)

**Authors**: Jakub Hoscilowicz, Pawel Popiolek, Jan Rudkowski, Jedrzej Bieniasz, Artur Janicki

**Abstract**: Simple fine-tuning can embed hidden text into large language models (LLMs), which is revealed only when triggered by a specific query. Applications include LLM fingerprinting, where a unique identifier is embedded to verify licensing compliance, and steganography, where the LLM carries hidden messages disclosed through a trigger query.   Our work demonstrates that embedding hidden text via fine-tuning, although seemingly secure due to the vast number of potential triggers, is vulnerable to extraction through analysis of the LLM's output decoding process. We introduce an extraction attack called Unconditional Token Forcing (UTF), which iteratively feeds tokens from the LLM's vocabulary to reveal sequences with high token probabilities, indicating hidden text candidates. We also present Unconditional Token Forcing Confusion (UTFC), a defense paradigm that makes hidden text resistant to all known extraction attacks without degrading the general performance of LLMs compared to standard fine-tuning. UTFC has both benign (improving LLM fingerprinting) and malign applications (using LLMs to create covert communication channels).



## **5. Privacy Evaluation Benchmarks for NLP Models**

cs.CL

Accepted by the Findings of EMNLP 2024

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15868v1) [paper-pdf](http://arxiv.org/pdf/2409.15868v1)

**Authors**: Wei Huang, Yinggui Wang, Cen Chen

**Abstract**: By inducing privacy attacks on NLP models, attackers can obtain sensitive information such as training data and model parameters, etc. Although researchers have studied, in-depth, several kinds of attacks in NLP models, they are non-systematic analyses. It lacks a comprehensive understanding of the impact caused by the attacks. For example, we must consider which scenarios can apply to which attacks, what the common factors are that affect the performance of different attacks, the nature of the relationships between different attacks, and the influence of various datasets and models on the effectiveness of the attacks, etc. Therefore, we need a benchmark to holistically assess the privacy risks faced by NLP models. In this paper, we present a privacy attack and defense evaluation benchmark in the field of NLP, which includes the conventional/small models and large language models (LLMs). This benchmark supports a variety of models, datasets, and protocols, along with standardized modules for comprehensive evaluation of attacks and defense strategies. Based on the above framework, we present a study on the association between auxiliary data from different domains and the strength of privacy attacks. And we provide an improved attack method in this scenario with the help of Knowledge Distillation (KD). Furthermore, we propose a chained framework for privacy attacks. Allowing a practitioner to chain multiple attacks to achieve a higher-level attack objective. Based on this, we provide some defense and enhanced attack strategies. The code for reproducing the results can be found at https://github.com/user2311717757/nlp_doctor.



## **6. Goal-guided Generative Prompt Injection Attack on Large Language Models**

cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2404.07234v3) [paper-pdf](http://arxiv.org/pdf/2404.07234v3)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.



## **7. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

cs.CL

Accepted to the main conference of EMNLP 2024

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2407.00869v2) [paper-pdf](http://arxiv.org/pdf/2407.00869v2)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.



## **8. Uncovering Coordinated Cross-Platform Information Operations Threatening the Integrity of the 2024 U.S. Presidential Election Online Discussion**

cs.SI

The 2024 Election Integrity Initiative: HUMANS Lab - Working Paper  No. 2024.4 - University of Southern California

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15402v1) [paper-pdf](http://arxiv.org/pdf/2409.15402v1)

**Authors**: Marco Minici, Luca Luceri, Federico Cinus, Emilio Ferrara

**Abstract**: Information Operations (IOs) pose a significant threat to the integrity of democratic processes, with the potential to influence election-related online discourse. In anticipation of the 2024 U.S. presidential election, we present a study aimed at uncovering the digital traces of coordinated IOs on $\mathbb{X}$ (formerly Twitter). Using our machine learning framework for detecting online coordination, we analyze a dataset comprising election-related conversations on $\mathbb{X}$ from May 2024. This reveals a network of coordinated inauthentic actors, displaying notable similarities in their link-sharing behaviors. Our analysis shows concerted efforts by these accounts to disseminate misleading, redundant, and biased information across the Web through a coordinated cross-platform information operation: The links shared by this network frequently direct users to other social media platforms or suspicious websites featuring low-quality political content and, in turn, promoting the same $\mathbb{X}$ and YouTube accounts. Members of this network also shared deceptive images generated by AI, accompanied by language attacking political figures and symbolic imagery intended to convey power and dominance. While $\mathbb{X}$ has suspended a subset of these accounts, more than 75% of the coordinated network remains active. Our findings underscore the critical role of developing computational models to scale up the detection of threats on large social media platforms, and emphasize the broader implications of these techniques to detect IOs across the wider Web.



## **9. Membership Inference Attacks and Privacy in Topic Modeling**

cs.CR

13 pages + appendices and references. 9 figures

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2403.04451v2) [paper-pdf](http://arxiv.org/pdf/2403.04451v2)

**Authors**: Nico Manzonelli, Wanrong Zhang, Salil Vadhan

**Abstract**: Recent research shows that large language models are susceptible to privacy attacks that infer aspects of the training data. However, it is unclear if simpler generative models, like topic models, share similar vulnerabilities. In this work, we propose an attack against topic models that can confidently identify members of the training data in Latent Dirichlet Allocation. Our results suggest that the privacy risks associated with generative modeling are not restricted to large neural models. Additionally, to mitigate these vulnerabilities, we explore differentially private (DP) topic modeling. We propose a framework for private topic modeling that incorporates DP vocabulary selection as a pre-processing step, and show that it improves privacy while having limited effects on practical utility.



## **10. LLM in the Shell: Generative Honeypots**

cs.CR

6 pages. 2 figures. 2 tables

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2309.00155v3) [paper-pdf](http://arxiv.org/pdf/2309.00155v3)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: Honeypots are essential tools in cybersecurity for early detection, threat intelligence gathering, and analysis of attacker's behavior. However, most of them lack the required realism to engage and fool human attackers long-term. Being easy to distinguish honeypots strongly hinders their effectiveness. This can happen because they are too deterministic, lack adaptability, or lack deepness. This work introduces shelLM, a dynamic and realistic software honeypot based on Large Language Models that generates Linux-like shell output. We designed and implemented shelLM using cloud-based LLMs. We evaluated if shelLM can generate output as expected from a real Linux shell. The evaluation was done by asking cybersecurity researchers to use the honeypot and give feedback if each answer from the honeypot was the expected one from a Linux shell. Results indicate that shelLM can create credible and dynamic answers capable of addressing the limitations of current honeypots. ShelLM reached a TNR of 0.90, convincing humans it was consistent with a real Linux shell. The source code and prompts for replicating the experiments have been publicly available.



## **11. Attack Atlas: A Practitioner's Perspective on Challenges and Pitfalls in Red Teaming GenAI**

cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15398v1) [paper-pdf](http://arxiv.org/pdf/2409.15398v1)

**Authors**: Ambrish Rawat, Stefan Schoepf, Giulio Zizzo, Giandomenico Cornacchia, Muhammad Zaid Hameed, Kieran Fraser, Erik Miehling, Beat Buesser, Elizabeth M. Daly, Mark Purcell, Prasanna Sattigeri, Pin-Yu Chen, Kush R. Varshney

**Abstract**: As generative AI, particularly large language models (LLMs), become increasingly integrated into production applications, new attack surfaces and vulnerabilities emerge and put a focus on adversarial threats in natural language and multi-modal systems. Red-teaming has gained importance in proactively identifying weaknesses in these systems, while blue-teaming works to protect against such adversarial attacks. Despite growing academic interest in adversarial risks for generative AI, there is limited guidance tailored for practitioners to assess and mitigate these challenges in real-world environments. To address this, our contributions include: (1) a practical examination of red- and blue-teaming strategies for securing generative AI, (2) identification of key challenges and open questions in defense development and evaluation, and (3) the Attack Atlas, an intuitive framework that brings a practical approach to analyzing single-turn input attacks, placing it at the forefront for practitioners. This work aims to bridge the gap between academic insights and practical security measures for the protection of generative AI systems.



## **12. Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs**

cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14866v1) [paper-pdf](http://arxiv.org/pdf/2409.14866v1)

**Authors**: Xueluan Gong, Mingzhe Li, Yilin Zhang, Fengyuan Ran, Chen Chen, Yanjiao Chen, Qian Wang, Kwok-Yan Lam

**Abstract**: Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates, our method starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks.   We evaluated our method on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, our method achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60%. Additionally, our method can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, our method can achieve over 78\% attack success rate even with 100 tokens. Moreover, our method demonstrates transferability and is robust to state-of-the-art defenses. We will open-source our codes upon publication.



## **13. A Security Risk Taxonomy for Prompt-Based Interaction With Large Language Models**

cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2311.11415v2) [paper-pdf](http://arxiv.org/pdf/2311.11415v2)

**Authors**: Erik Derner, Kristina Batistič, Jan Zahálka, Robert Babuška

**Abstract**: As large language models (LLMs) permeate more and more applications, an assessment of their associated security risks becomes increasingly necessary. The potential for exploitation by malicious actors, ranging from disinformation to data breaches and reputation damage, is substantial. This paper addresses a gap in current research by specifically focusing on security risks posed by LLMs within the prompt-based interaction scheme, which extends beyond the widely covered ethical and societal implications. Our work proposes a taxonomy of security risks along the user-model communication pipeline and categorizes the attacks by target and attack type alongside the commonly used confidentiality, integrity, and availability (CIA) triad. The taxonomy is reinforced with specific attack examples to showcase the real-world impact of these risks. Through this taxonomy, we aim to inform the development of robust and secure LLM applications, enhancing their safety and trustworthiness.



## **14. LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation**

eess.AS

5 pages, submitted to ICASSP 2025

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14743v1) [paper-pdf](http://arxiv.org/pdf/2409.14743v1)

**Authors**: Hieu-Thi Luong, Haoyang Li, Lin Zhang, Kong Aik Lee, Eng Siong Chng

**Abstract**: Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining information valuable to both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.44% equal error rate.



## **15. PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs**

cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14729v1) [paper-pdf](http://arxiv.org/pdf/2409.14729v1)

**Authors**: Jiahao Yu, Yangguang Shao, Hanwen Miao, Junzheng Shi, Xinyu Xing

**Abstract**: Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks.   In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts.   By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.



## **16. Enhancing LLM-based Autonomous Driving Agents to Mitigate Perception Attacks**

cs.CR

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2409.14488v1) [paper-pdf](http://arxiv.org/pdf/2409.14488v1)

**Authors**: Ruoyu Song, Muslum Ozgur Ozmen, Hyungsub Kim, Antonio Bianchi, Z. Berkay Celik

**Abstract**: There is a growing interest in integrating Large Language Models (LLMs) with autonomous driving (AD) systems. However, AD systems are vulnerable to attacks against their object detection and tracking (ODT) functions. Unfortunately, our evaluation of four recent LLM agents against ODT attacks shows that the attacks are 63.26% successful in causing them to crash or violate traffic rules due to (1) misleading memory modules that provide past experiences for decision making, (2) limitations of prompts in identifying inconsistencies, and (3) reliance on ground truth perception data.   In this paper, we introduce Hudson, a driving reasoning agent that extends prior LLM-based driving systems to enable safer decision making during perception attacks while maintaining effectiveness under benign conditions. Hudson achieves this by first instrumenting the AD software to collect real-time perception results and contextual information from the driving scene. This data is then formalized into a domain-specific language (DSL). To guide the LLM in detecting and making safe control decisions during ODT attacks, Hudson translates the DSL into natural language, along with a list of custom attack detection instructions. Following query execution, Hudson analyzes the LLM's control decision to understand its causal reasoning process.   We evaluate the effectiveness of Hudson using a proprietary LLM (GPT-4) and two open-source LLMs (Llama and Gemma) in various adversarial driving scenarios. GPT-4, Llama, and Gemma achieve, on average, an attack detection accuracy of 83. 3%, 63. 6%, and 73. 6%. Consequently, they make safe control decisions in 86.4%, 73.9%, and 80% of the attacks. Our results, following the growing interest in integrating LLMs into AD systems, highlight the strengths of LLMs and their potential to detect and mitigate ODT attacks.



## **17. PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach**

cs.CR

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.14177v1) [paper-pdf](http://arxiv.org/pdf/2409.14177v1)

**Authors**: Zhihao Lin, Wei Ma, Mingyi Zhou, Yanjie Zhao, Haoyu Wang, Yang Liu, Jun Wang, Li Li

**Abstract**: In recent years, Large Language Models (LLMs) have gained widespread use, accompanied by increasing concerns over their security. Traditional jailbreak attacks rely on internal model details or have limitations when exploring the unsafe behavior of the victim model, limiting their generalizability. In this paper, we introduce PathSeeker, a novel black-box jailbreak method inspired by the concept of escaping a security maze. This work is inspired by the game of rats escaping a maze. We think that each LLM has its unique "security maze", and attackers attempt to find the exit learning from the received feedback and their accumulated experience to compromise the target LLM's security defences. Our approach leverages multi-agent reinforcement learning, where smaller models collaborate to guide the main LLM in performing mutation operations to achieve the attack objectives. By progressively modifying inputs based on the model's feedback, our system induces richer, harmful responses. During our manual attempts to perform jailbreak attacks, we found that the vocabulary of the response of the target model gradually became richer and eventually produced harmful responses. Based on the observation, we also introduce a reward mechanism that exploits the expansion of vocabulary richness in LLM responses to weaken security constraints. Our method outperforms five state-of-the-art attack techniques when tested across 13 commercial and open-source LLMs, achieving high attack success rates, especially in strongly aligned commercial models like GPT-4o-mini, Claude-3.5, and GLM-4-air with strong safety alignment. This study aims to improve the understanding of LLM security vulnerabilities and we hope that this sturdy can contribute to the development of more robust defenses.



## **18. Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm**

cs.CL

Under Review

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.14119v1) [paper-pdf](http://arxiv.org/pdf/2409.14119v1)

**Authors**: Jaehan Kim, Minkyoo Song, Seung Ho Na, Seungwon Shin

**Abstract**: Parameter-efficient fine-tuning (PEFT) has become a key training strategy for large language models. However, its reliance on fewer trainable parameters poses security risks, such as task-agnostic backdoors. Despite their severe impact on a wide range of tasks, there is no practical defense solution available that effectively counters task-agnostic backdoors within the context of PEFT. In this study, we introduce Obliviate, a PEFT-integrable backdoor defense. We develop two techniques aimed at amplifying benign neurons within PEFT layers and penalizing the influence of trigger tokens. Our evaluations across three major PEFT architectures show that our method can significantly reduce the attack success rate of the state-of-the-art task-agnostic backdoors (83.6%$\downarrow$). Furthermore, our method exhibits robust defense capabilities against both task-specific backdoors and adaptive attacks. Source code will be obtained at https://github.com/obliviateARR/Obliviate.



## **19. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2407.14573v6) [paper-pdf](http://arxiv.org/pdf/2407.14573v6)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **20. ConfusedPilot: Confused Deputy Risks in RAG-based LLMs**

cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.04870v4) [paper-pdf](http://arxiv.org/pdf/2408.04870v4)

**Authors**: Ayush RoyChowdhury, Mulong Luo, Prateek Sahu, Sarbartha Banerjee, Mohit Tiwari

**Abstract**: Retrieval augmented generation (RAG) is a process where a large language model (LLM) retrieves useful information from a database and then generates the responses. It is becoming popular in enterprise settings for daily business operations. For example, Copilot for Microsoft 365 has accumulated millions of businesses. However, the security implications of adopting such RAG-based systems are unclear.   In this paper, we introduce ConfusedPilot, a class of security vulnerabilities of RAG systems that confuse Copilot and cause integrity and confidentiality violations in its responses. First, we investigate a vulnerability that embeds malicious text in the modified prompt in RAG, corrupting the responses generated by the LLM. Second, we demonstrate a vulnerability that leaks secret data, which leverages the caching mechanism during retrieval. Third, we investigate how both vulnerabilities can be exploited to propagate misinformation within the enterprise and ultimately impact its operations, such as sales and manufacturing. We also discuss the root cause of these attacks by investigating the architecture of a RAG-based system. This study highlights the security vulnerabilities in today's RAG-based systems and proposes design guidelines to secure future RAG-based systems.



## **21. On the Feasibility of Fully AI-automated Vishing Attacks**

cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13793v1) [paper-pdf](http://arxiv.org/pdf/2409.13793v1)

**Authors**: João Figueiredo, Afonso Carvalho, Daniel Castro, Daniel Gonçalves, Nuno Santos

**Abstract**: A vishing attack is a form of social engineering where attackers use phone calls to deceive individuals into disclosing sensitive information, such as personal data, financial information, or security credentials. Attackers exploit the perceived urgency and authenticity of voice communication to manipulate victims, often posing as legitimate entities like banks or tech support. Vishing is a particularly serious threat as it bypasses security controls designed to protect information. In this work, we study the potential for vishing attacks to escalate with the advent of AI. In theory, AI-powered software bots may have the ability to automate these attacks by initiating conversations with potential victims via phone calls and deceiving them into disclosing sensitive information. To validate this thesis, we introduce ViKing, an AI-powered vishing system developed using publicly available AI technology. It relies on a Large Language Model (LLM) as its core cognitive processor to steer conversations with victims, complemented by a pipeline of speech-to-text and text-to-speech modules that facilitate audio-text conversion in phone calls. Through a controlled social experiment involving 240 participants, we discovered that ViKing has successfully persuaded many participants to reveal sensitive information, even those who had been explicitly warned about the risk of vishing campaigns. Interactions with ViKing's bots were generally considered realistic. From these findings, we conclude that tools like ViKing may already be accessible to potential malicious actors, while also serving as an invaluable resource for cyber awareness programs.



## **22. Prompt Obfuscation for Large Language Models**

cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.11026v2) [paper-pdf](http://arxiv.org/pdf/2409.11026v2)

**Authors**: David Pape, Thorsten Eisenhofer, Lea Schönherr

**Abstract**: System prompts that include detailed instructions to describe the task performed by the underlying large language model (LLM) can easily transform foundation models into tools and services with minimal overhead. Because of their crucial impact on the utility, they are often considered intellectual property, similar to the code of a software product. However, extracting system prompts is easily possible by using prompt injection. As of today, there is no effective countermeasure to prevent the stealing of system prompts and all safeguarding efforts could be evaded with carefully crafted prompt injections that bypass all protection mechanisms. In this work, we propose an alternative to conventional system prompts. We introduce prompt obfuscation to prevent the extraction of the system prompt while maintaining the utility of the system itself with only little overhead. The core idea is to find a representation of the original system prompt that leads to the same functionality, while the obfuscated system prompt does not contain any information that allows conclusions to be drawn about the original system prompt. We implement an optimization-based method to find an obfuscated prompt representation while maintaining the functionality. To evaluate our approach, we investigate eight different metrics to compare the performance of a system using the original and the obfuscated system prompts, and we show that the obfuscated version is constantly on par with the original one. We further perform three different deobfuscation attacks and show that with access to the obfuscated prompt and the LLM itself, we are not able to consistently extract meaningful information. Overall, we showed that prompt obfuscation can be an effective method to protect intellectual property while maintaining the same utility as the original system prompt.



## **23. Applying Pre-trained Multilingual BERT in Embeddings for Improved Malicious Prompt Injection Attacks Detection**

cs.CL

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13331v1) [paper-pdf](http://arxiv.org/pdf/2409.13331v1)

**Authors**: Md Abdur Rahman, Hossain Shahriar, Fan Wu, Alfredo Cuzzocrea

**Abstract**: Large language models (LLMs) are renowned for their exceptional capabilities, and applying to a wide range of applications. However, this widespread use brings significant vulnerabilities. Also, it is well observed that there are huge gap which lies in the need for effective detection and mitigation strategies against malicious prompt injection attacks in large language models, as current approaches may not adequately address the complexity and evolving nature of these vulnerabilities in real-world applications. Therefore, this work focuses the impact of malicious prompt injection attacks which is one of most dangerous vulnerability on real LLMs applications. It examines to apply various BERT (Bidirectional Encoder Representations from Transformers) like multilingual BERT, DistilBert for classifying malicious prompts from legitimate prompts. Also, we observed how tokenizing the prompt texts and generating embeddings using multilingual BERT contributes to improve the performance of various machine learning methods: Gaussian Naive Bayes, Random Forest, Support Vector Machine, and Logistic Regression. The performance of each model is rigorously analyzed with various parameters to improve the binary classification to discover malicious prompts. Multilingual BERT approach to embed the prompts significantly improved and outperformed the existing works and achieves an outstanding accuracy of 96.55% by Logistic regression. Additionally, we investigated the incorrect predictions of the model to gain insights into its limitations. The findings can guide researchers in tuning various BERT for finding the most suitable model for diverse LLMs vulnerabilities.



## **24. An Adaptive End-to-End IoT Security Framework Using Explainable AI and LLMs**

cs.LG

6 pages, 1 figure, Accepted in 2024 IEEE WF-IoT Conference

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13177v1) [paper-pdf](http://arxiv.org/pdf/2409.13177v1)

**Authors**: Sudipto Baral, Sajal Saha, Anwar Haque

**Abstract**: The exponential growth of the Internet of Things (IoT) has significantly increased the complexity and volume of cybersecurity threats, necessitating the development of advanced, scalable, and interpretable security frameworks. This paper presents an innovative, comprehensive framework for real-time IoT attack detection and response that leverages Machine Learning (ML), Explainable AI (XAI), and Large Language Models (LLM). By integrating XAI techniques such as SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) with a model-independent architecture, we ensure our framework's adaptability across various ML algorithms. Additionally, the incorporation of LLMs enhances the interpretability and accessibility of detection decisions, providing system administrators with actionable, human-understandable explanations of detected threats. Our end-to-end framework not only facilitates a seamless transition from model development to deployment but also represents a real-world application capability that is often lacking in existing research. Based on our experiments with the CIC-IOT-2023 dataset \cite{neto2023ciciot2023}, Gemini and OPENAI LLMS demonstrate unique strengths in attack mitigation: Gemini offers precise, focused strategies, while OPENAI provides extensive, in-depth security measures. Incorporating SHAP and LIME algorithms within XAI provides comprehensive insights into attack detection, emphasizing opportunities for model improvement through detailed feature analysis, fine-tuning, and the adaptation of misclassifications to enhance accuracy.



## **25. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13174v1) [paper-pdf](http://arxiv.org/pdf/2409.13174v1)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical security threats.



## **26. Defending against Reverse Preference Attacks is Difficult**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.



## **27. Enhancing Jailbreak Attacks with Diversity Guidance**

cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2403.00292v2) [paper-pdf](http://arxiv.org/pdf/2403.00292v2)

**Authors**: Xu Zhang, Dinghao Jing, Xiaojun Wan

**Abstract**: As large language models(LLMs) become commonplace in practical applications, the security issues of LLMs have attracted societal concerns. Although extensive efforts have been made to safety alignment, LLMs remain vulnerable to jailbreak attacks. We find that redundant computations limit the performance of existing jailbreak attack methods. Therefore, we propose DPP-based Stochastic Trigger Searching (DSTS), a new optimization algorithm for jailbreak attacks. DSTS incorporates diversity guidance through techniques including stochastic gradient search and DPP selection during optimization. Detailed experiments and ablation studies demonstrate the effectiveness of the algorithm. Moreover, we use the proposed algorithm to compute the risk boundaries for different LLMs, providing a new perspective on LLM safety evaluation.



## **28. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2405.20090v2) [paper-pdf](http://arxiv.org/pdf/2405.20090v2)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jiahang Cao, Le Yang, Jize Zhang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.



## **29. Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model**

cs.CV

This paper is accepted by ECCV 2024

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.19150v3) [paper-pdf](http://arxiv.org/pdf/2402.19150v3)

**Authors**: Hao Cheng, Erjia Xiao, Jindong Gu, Le Yang, Jinhao Duan, Jize Zhang, Jiahang Cao, Kaidi Xu, Renjing Xu

**Abstract**: Large Vision-Language Models (LVLMs) rely on vision encoders and Large Language Models (LLMs) to exhibit remarkable capabilities on various multi-modal tasks in the joint space of vision and language. However, typographic attacks, which disrupt Vision-Language Models (VLMs) such as Contrastive Language-Image Pretraining (CLIP), have also been expected to be a security threat to LVLMs. Firstly, we verify typographic attacks on current well-known commercial and open-source LVLMs and uncover the widespread existence of this threat. Secondly, to better assess this vulnerability, we propose the most comprehensive and largest-scale Typographic Dataset to date. The Typographic Dataset not only considers the evaluation of typographic attacks under various multi-modal tasks but also evaluates the effects of typographic attacks, influenced by texts generated with diverse factors. Based on the evaluation results, we investigate the causes why typographic attacks impacting VLMs and LVLMs, leading to three highly insightful discoveries. During the process of further validating the rationality of our discoveries, we can reduce the performance degradation caused by typographic attacks from 42.07\% to 13.90\%. Code and Dataset are available in \href{https://github.com/ChaduCheng/TypoDeceptions}



## **30. LLM-Powered Text Simulation Attack Against ID-Free Recommender Systems**

cs.IR

12 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.11690v2) [paper-pdf](http://arxiv.org/pdf/2409.11690v2)

**Authors**: Zongwei Wang, Min Gao, Junliang Yu, Xinyi Gao, Quoc Viet Hung Nguyen, Shazia Sadiq, Hongzhi Yin

**Abstract**: The ID-free recommendation paradigm has been proposed to address the limitation that traditional recommender systems struggle to model cold-start users or items with new IDs. Despite its effectiveness, this study uncovers that ID-free recommender systems are vulnerable to the proposed Text Simulation attack (TextSimu) which aims to promote specific target items. As a novel type of text poisoning attack, TextSimu exploits large language models (LLM) to alter the textual information of target items by simulating the characteristics of popular items. It operates effectively in both black-box and white-box settings, utilizing two key components: a unified popularity extraction module, which captures the essential characteristics of popular items, and an N-persona consistency simulation strategy, which creates multiple personas to collaboratively synthesize refined promotional textual descriptions for target items by simulating the popular items. To withstand TextSimu-like attacks, we further explore the detection approach for identifying LLM-generated promotional text. Extensive experiments conducted on three datasets demonstrate that TextSimu poses a more significant threat than existing poisoning attacks, while our defense method can detect malicious text of target items generated by TextSimu. By identifying the vulnerability, we aim to advance the development of more robust ID-free recommender systems.



## **31. AirGapAgent: Protecting Privacy-Conscious Conversational Agents**

cs.CR

at CCS'24

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2405.05175v2) [paper-pdf](http://arxiv.org/pdf/2405.05175v2)

**Authors**: Eugene Bagdasarian, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.



## **32. Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards**

cs.RO

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.12274v1) [paper-pdf](http://arxiv.org/pdf/2409.12274v1)

**Authors**: Yuwei Wu, Yuezhan Tao, Peihan Li, Guangyao Shi, Gaurav S. Sukhatmem, Vijay Kumar, Lifeng Zhou

**Abstract**: In this paper, we propose a hierarchical Large Language Models (LLMs) in-the-loop optimization framework for real-time multi-robot task allocation and target tracking in an unknown hazardous environment subject to sensing and communication attacks. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. The inner LLM adjusts parameters to prioritize various objectives, including performance, safety, and energy efficiency, while the outer LLM handles online variable completion for team reconfiguration. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, which provide the potential for safe LLM integration for multi-robot problems.



## **33. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2407.20361v2) [paper-pdf](http://arxiv.org/pdf/2407.20361v2)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.



## **34. Multitask Mayhem: Unveiling and Mitigating Safety Gaps in LLMs Fine-tuning**

cs.CL

19 pages, 11 figures

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.15361v1) [paper-pdf](http://arxiv.org/pdf/2409.15361v1)

**Authors**: Essa Jan, Nouar AlDahoul, Moiz Ali, Faizan Ahmad, Fareed Zaffar, Yasir Zaki

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to their adoption across a wide range of tasks, ranging from code generation to machine translation and sentiment analysis, etc. Red teaming/Safety alignment efforts show that fine-tuning models on benign (non-harmful) data could compromise safety. However, it remains unclear to what extent this phenomenon is influenced by different variables, including fine-tuning task, model calibrations, etc. This paper explores the task-wise safety degradation due to fine-tuning on downstream tasks such as summarization, code generation, translation, and classification across various calibration. Our results reveal that: 1) Fine-tuning LLMs for code generation and translation leads to the highest degradation in safety guardrails. 2) LLMs generally have weaker guardrails for translation and classification, with 73-92% of harmful prompts answered, across baseline and other calibrations, falling into one of two concern categories. 3) Current solutions, including guards and safety tuning datasets, lack cross-task robustness. To address these issues, we developed a new multitask safety dataset effectively reducing attack success rates across a range of tasks without compromising the model's overall helpfulness. Our work underscores the need for generalized alignment measures to ensure safer and more robust models.



## **35. DrLLM: Prompt-Enhanced Distributed Denial-of-Service Resistance Method with Large Language Models**

cs.CR

**SubmitDate**: 2024-09-18    [abs](http://arxiv.org/abs/2409.10561v2) [paper-pdf](http://arxiv.org/pdf/2409.10561v2)

**Authors**: Zhenyu Yin, Shang Liu, Guangyuan Xu

**Abstract**: The increasing number of Distributed Denial of Service (DDoS) attacks poses a major threat to the Internet, highlighting the importance of DDoS mitigation. Most existing approaches require complex training methods to learn data features, which increases the complexity and generality of the application. In this paper, we propose DrLLM, which aims to mine anomalous traffic information in zero-shot scenarios through Large Language Models (LLMs). To bridge the gap between DrLLM and existing approaches, we embed the global and local information of the traffic data into the reasoning paradigm and design three modules, namely Knowledge Embedding, Token Embedding, and Progressive Role Reasoning, for data representation and reasoning. In addition we explore the generalization of prompt engineering in the cybersecurity domain to improve the classification capability of DrLLM. Our ablation experiments demonstrate the applicability of DrLLM in zero-shot scenarios and further demonstrate the potential of LLMs in the network domains. DrLLM implementation code has been open-sourced at https://github.com/liuup/DrLLM.



## **36. Jailbreaking Large Language Models with Symbolic Mathematics**

cs.CR

**SubmitDate**: 2024-09-17    [abs](http://arxiv.org/abs/2409.11445v1) [paper-pdf](http://arxiv.org/pdf/2409.11445v1)

**Authors**: Emet Bethany, Mazal Bethany, Juan Arturo Nolazco Flores, Sumit Kumar Jha, Peyman Najafirad

**Abstract**: Recent advancements in AI safety have led to increased efforts in training and red-teaming large language models (LLMs) to mitigate unsafe content generation. However, these safety mechanisms may not be comprehensive, leaving potential vulnerabilities unexplored. This paper introduces MathPrompt, a novel jailbreaking technique that exploits LLMs' advanced capabilities in symbolic mathematics to bypass their safety mechanisms. By encoding harmful natural language prompts into mathematical problems, we demonstrate a critical vulnerability in current AI safety measures. Our experiments across 13 state-of-the-art LLMs reveal an average attack success rate of 73.6\%, highlighting the inability of existing safety training mechanisms to generalize to mathematically encoded inputs. Analysis of embedding vectors shows a substantial semantic shift between original and encoded prompts, helping explain the attack's success. This work emphasizes the importance of a holistic approach to AI safety, calling for expanded red-teaming efforts to develop robust safeguards across all potential input types and their associated risks.



## **37. Assessing biomedical knowledge robustness in large language models by query-efficient sampling attacks**

cs.CL

28 pages incl. appendix, updated version

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.10527v2) [paper-pdf](http://arxiv.org/pdf/2402.10527v2)

**Authors**: R. Patrick Xian, Alex J. Lee, Satvik Lolla, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. Understanding model vulnerabilities in high-stakes and knowledge-intensive tasks is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples (i.e. adversarial entities) in natural language processing tasks raises questions about their potential impact on the knowledge robustness of pre-trained and finetuned LLMs in high-stakes and specialized domains. We examined the use of type-consistent entity substitution as a template for collecting adversarial entities for billion-parameter LLMs with biomedical knowledge. To this end, we developed an embedding-space attack based on powerscaled distance-weighted sampling to assess the robustness of their biomedical knowledge with a low query budget and controllable coverage. Our method has favorable query efficiency and scaling over alternative approaches based on random sampling and blackbox gradient-guided search, which we demonstrated for adversarial distractor generation in biomedical question answering. Subsequent failure mode analysis uncovered two regimes of adversarial entities on the attack surface with distinct characteristics and we showed that entity substitution attacks can manipulate token-wise Shapley value explanations, which become deceptive in this setting. Our approach complements standard evaluations for high-capacity models and the results highlight the brittleness of domain knowledge in LLMs.



## **38. Security Attacks on LLM-based Code Completion Tools**

cs.CL

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2408.11006v2) [paper-pdf](http://arxiv.org/pdf/2408.11006v2)

**Authors**: Wen Cheng, Ke Sun, Xinyu Zhang, Wei Wang

**Abstract**: The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.



## **39. Do Membership Inference Attacks Work on Large Language Models?**

cs.CL

Accepted at Conference on Language Modeling (COLM), 2024

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2402.07841v2) [paper-pdf](http://arxiv.org/pdf/2402.07841v2)

**Authors**: Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, Hannaneh Hajishirzi

**Abstract**: Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn from the seemingly identical domain but with different temporal ranges. We release our code and data as a unified benchmark package that includes all existing MIAs, supporting future work.



## **40. Robust Privacy Amidst Innovation with Large Language Models Through a Critical Assessment of the Risks**

cs.CL

13 pages, 4 figures, 1 table, 1 supplementary, under review

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2407.16166v2) [paper-pdf](http://arxiv.org/pdf/2407.16166v2)

**Authors**: Yao-Shun Chuang, Atiquer Rahman Sarkar, Yu-Chun Hsu, Noman Mohammed, Xiaoqian Jiang

**Abstract**: This study examines integrating EHRs and NLP with large language models (LLMs) to improve healthcare data management and patient care. It focuses on using advanced models to create secure, HIPAA-compliant synthetic patient notes for biomedical research. The study used de-identified and re-identified MIMIC III datasets with GPT-3.5, GPT-4, and Mistral 7B to generate synthetic notes. Text generation employed templates and keyword extraction for contextually relevant notes, with one-shot generation for comparison. Privacy assessment checked PHI occurrence, while text utility was tested using an ICD-9 coding task. Text quality was evaluated with ROUGE and cosine similarity metrics to measure semantic similarity with source notes. Analysis of PHI occurrence and text utility via the ICD-9 coding task showed that the keyword-based method had low risk and good performance. One-shot generation showed the highest PHI exposure and PHI co-occurrence, especially in geographic location and date categories. The Normalized One-shot method achieved the highest classification accuracy. Privacy analysis revealed a critical balance between data utility and privacy protection, influencing future data use and sharing. Re-identified data consistently outperformed de-identified data. This study demonstrates the effectiveness of keyword-based methods in generating privacy-protecting synthetic clinical notes that retain data usability, potentially transforming clinical data-sharing practices. The superior performance of re-identified over de-identified data suggests a shift towards methods that enhance utility and privacy by using dummy PHIs to perplex privacy attacks.



## **41. LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses**

cs.CR

**SubmitDate**: 2024-09-16    [abs](http://arxiv.org/abs/2406.04755v2) [paper-pdf](http://arxiv.org/pdf/2406.04755v2)

**Authors**: Weiran Lin, Anna Gerchanovsky, Omer Akgul, Lujo Bauer, Matt Fredrikson, Zifan Wang

**Abstract**: Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.



## **42. LLM Honeypot: Leveraging Large Language Models as Advanced Interactive Honeypot Systems**

cs.CR

6 pages, 5 figures

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.08234v2) [paper-pdf](http://arxiv.org/pdf/2409.08234v2)

**Authors**: Hakan T. Otal, M. Abdullah Canbaz

**Abstract**: The rapid evolution of cyber threats necessitates innovative solutions for detecting and analyzing malicious activity. Honeypots, which are decoy systems designed to lure and interact with attackers, have emerged as a critical component in cybersecurity. In this paper, we present a novel approach to creating realistic and interactive honeypot systems using Large Language Models (LLMs). By fine-tuning a pre-trained open-source language model on a diverse dataset of attacker-generated commands and responses, we developed a honeypot capable of sophisticated engagement with attackers. Our methodology involved several key steps: data collection and processing, prompt engineering, model selection, and supervised fine-tuning to optimize the model's performance. Evaluation through similarity metrics and live deployment demonstrated that our approach effectively generates accurate and informative responses. The results highlight the potential of LLMs to revolutionize honeypot technology, providing cybersecurity professionals with a powerful tool to detect and analyze malicious activity, thereby enhancing overall security infrastructure.



## **43. Detection Made Easy: Potentials of Large Language Models for Solidity Vulnerabilities**

cs.CR

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.10574v1) [paper-pdf](http://arxiv.org/pdf/2409.10574v1)

**Authors**: Md Tauseef Alam, Raju Halder, Abyayananda Maiti

**Abstract**: The large-scale deployment of Solidity smart contracts on the Ethereum mainnet has increasingly attracted financially-motivated attackers in recent years. A few now-infamous attacks in Ethereum's history includes DAO attack in 2016 (50 million dollars lost), Parity Wallet hack in 2017 (146 million dollars locked), Beautychain's token BEC in 2018 (900 million dollars market value fell to 0), and NFT gaming blockchain breach in 2022 ($600 million in Ether stolen). This paper presents a comprehensive investigation of the use of large language models (LLMs) and their capabilities in detecting OWASP Top Ten vulnerabilities in Solidity. We introduce a novel, class-balanced, structured, and labeled dataset named VulSmart, which we use to benchmark and compare the performance of open-source LLMs such as CodeLlama, Llama2, CodeT5 and Falcon, alongside closed-source models like GPT-3.5 Turbo and GPT-4o Mini. Our proposed SmartVD framework is rigorously tested against these models through extensive automated and manual evaluations, utilizing BLEU and ROUGE metrics to assess the effectiveness of vulnerability detection in smart contracts. We also explore three distinct prompting strategies-zero-shot, few-shot, and chain-of-thought-to evaluate the multi-class classification and generative capabilities of the SmartVD framework. Our findings reveal that SmartVD outperforms its open-source counterparts and even exceeds the performance of closed-source base models like GPT-3.5 and GPT-4 Mini. After fine-tuning, the closed-source models, GPT-3.5 Turbo and GPT-4o Mini, achieved remarkable performance with 99% accuracy in detecting vulnerabilities, 94% in identifying their types, and 98% in determining severity. Notably, SmartVD performs best with the `chain-of-thought' prompting technique, whereas the fine-tuned closed-source models excel with the `zero-shot' prompting approach.



## **44. ContractTinker: LLM-Empowered Vulnerability Repair for Real-World Smart Contracts**

cs.SE

4 pages, and to be accepted in ASE2024

**SubmitDate**: 2024-09-15    [abs](http://arxiv.org/abs/2409.09661v1) [paper-pdf](http://arxiv.org/pdf/2409.09661v1)

**Authors**: Che Wang, Jiashuo Zhang, Jianbo Gao, Libin Xia, Zhi Guan, Zhong Chen

**Abstract**: Smart contracts are susceptible to being exploited by attackers, especially when facing real-world vulnerabilities. To mitigate this risk, developers often rely on third-party audit services to identify potential vulnerabilities before project deployment. Nevertheless, repairing the identified vulnerabilities is still complex and labor-intensive, particularly for developers lacking security expertise. Moreover, existing pattern-based repair tools mostly fail to address real-world vulnerabilities due to their lack of high-level semantic understanding. To fill this gap, we propose ContractTinker, a Large Language Models (LLMs)-empowered tool for real-world vulnerability repair. The key insight is our adoption of the Chain-of-Thought approach to break down the entire generation task into sub-tasks. Additionally, to reduce hallucination, we integrate program static analysis to guide the LLM. We evaluate ContractTinker on 48 high-risk vulnerabilities. The experimental results show that among the patches generated by ContractTinker, 23 (48%) are valid patches that fix the vulnerabilities, while 10 (21%) require only minor modifications. A video of ContractTinker is available at https://youtu.be/HWFVi-YHcPE.



## **45. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.04585v3) [paper-pdf](http://arxiv.org/pdf/2408.04585v3)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs and conduct extensive experiments on three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.



## **46. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-09-14    [abs](http://arxiv.org/abs/2408.00761v3) [paper-pdf](http://arxiv.org/pdf/2408.00761v3)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **47. Safeguarding AI Agents: Developing and Analyzing Safety Architectures**

cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2409.03793v2) [paper-pdf](http://arxiv.org/pdf/2409.03793v2)

**Authors**: Ishaan Domkundwar, Mukunda N S, Ishaan Bhola

**Abstract**: AI agents, specifically powered by large language models, have demonstrated exceptional capabilities in various applications where precision and efficacy are necessary. However, these agents come with inherent risks, including the potential for unsafe or biased actions, vulnerability to adversarial attacks, lack of transparency, and tendency to generate hallucinations. As AI agents become more prevalent in critical sectors of the industry, the implementation of effective safety protocols becomes increasingly important. This paper addresses the critical need for safety measures in AI systems, especially ones that collaborate with human teams. We propose and evaluate three frameworks to enhance safety protocols in AI agent systems: an LLM-powered input-output filter, a safety agent integrated within the system, and a hierarchical delegation-based system with embedded safety checks. Our methodology involves implementing these frameworks and testing them against a set of unsafe agentic use cases, providing a comprehensive evaluation of their effectiveness in mitigating risks associated with AI agent deployment. We conclude that these frameworks can significantly strengthen the safety and security of AI agent systems, minimizing potential harmful actions or outputs. Our work contributes to the ongoing effort to create safe and reliable AI applications, particularly in automated operations, and provides a foundation for developing robust guardrails to ensure the responsible use of AI agents in real-world applications.



## **48. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

cs.CR

**SubmitDate**: 2024-09-13    [abs](http://arxiv.org/abs/2408.04811v2) [paper-pdf](http://arxiv.org/pdf/2408.04811v2)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.



## **49. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

cs.CL

Oral presentation at KDD 2024 GenAI Evaluation workshop

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2405.02764v2) [paper-pdf](http://arxiv.org/pdf/2405.02764v2)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.



## **50. Securing Large Language Models: Addressing Bias, Misinformation, and Prompt Attacks**

cs.CR

17 pages, 1 figure

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08087v1) [paper-pdf](http://arxiv.org/pdf/2409.08087v1)

**Authors**: Benji Peng, Keyu Chen, Ming Li, Pohsun Feng, Ziqian Bi, Junyu Liu, Qian Niu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across various fields, yet their increasing use raises critical security concerns. This article reviews recent literature addressing key issues in LLM security, with a focus on accuracy, bias, content detection, and vulnerability to attacks. Issues related to inaccurate or misleading outputs from LLMs is discussed, with emphasis on the implementation from fact-checking methodologies to enhance response reliability. Inherent biases within LLMs are critically examined through diverse evaluation techniques, including controlled input studies and red teaming exercises. A comprehensive analysis of bias mitigation strategies is presented, including approaches from pre-processing interventions to in-training adjustments and post-processing refinements. The article also probes the complexity of distinguishing LLM-generated content from human-produced text, introducing detection mechanisms like DetectGPT and watermarking techniques while noting the limitations of machine learning enabled classifiers under intricate circumstances. Moreover, LLM vulnerabilities, including jailbreak attacks and prompt injection exploits, are analyzed by looking into different case studies and large-scale competitions like HackAPrompt. This review is concluded by retrospecting defense mechanisms to safeguard LLMs, accentuating the need for more extensive research into the LLM security field.



