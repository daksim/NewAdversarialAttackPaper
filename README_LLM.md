# Latest Adversarial Attack Papers
**update at 2023-09-18 09:33:32**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer**

cs.CL

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08583v1) [paper-pdf](http://arxiv.org/pdf/2309.08583v1)

**Authors**: Arkadiy Saakyan, Smaranda Muresan

**Abstract**: While state-of-the-art language models excel at the style transfer task, current work does not address explainability of style transfer systems. Explanations could be generated using large language models such as GPT-3.5 and GPT-4, but the use of such complex systems is inefficient when smaller, widely distributed, and transparent alternatives are available. We propose a framework to augment and improve a formality style transfer dataset with explanations via model distillation from ChatGPT. To further refine the generated explanations, we propose a novel way to incorporate scarce expert human feedback using in-context learning (ICLEF: In-Context Learning from Expert Feedback) by prompting ChatGPT to act as a critic to its own outputs. We use the resulting dataset of 9,960 explainable formality style transfer instances (e-GYAFC) to show that current openly distributed instruction-tuned models (and, in some settings, ChatGPT) perform poorly on the task, and that fine-tuning on our high-quality dataset leads to significant improvements as shown by automatic evaluation. In human evaluation, we show that models much smaller than ChatGPT fine-tuned on our data align better with expert preferences. Finally, we discuss two potential applications of models fine-tuned on the explainable style transfer task: interpretable authorship verification and interpretable adversarial attacks on AI-generated text detectors.



## **2. RAIN: Your Language Models Can Align Themselves without Finetuning**

cs.CL

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07124v1) [paper-pdf](http://arxiv.org/pdf/2309.07124v1)

**Authors**: Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

**Abstract**: Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%.



## **3. Games and Argumentation: Time for a Family Reunion!**

cs.LO

Fourth Workshop on Explainable Logic-Based Knowledge Representation  (XLoKR), Sept 2, 2023. Rhodes, Greece

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.06620v1) [paper-pdf](http://arxiv.org/pdf/2309.06620v1)

**Authors**: Bertram Ludäscher, Yilin Xia

**Abstract**: The rule "defeated(X) $\leftarrow$ attacks(Y,X), $\neg$ defeated(Y)" states that an argument is defeated if it is attacked by an argument that is not defeated. The rule "win(X) $\leftarrow$ move(X,Y), $\neg$ win(Y)" states that in a game a position is won if there is a move to a position that is not won. Both logic rules can be seen as close relatives (even identical twins) and both rules have been at the center of attention at various times in different communities: The first rule lies at the core of argumentation frameworks and has spawned a large family of models and semantics of abstract argumentation. The second rule has played a key role in the quest to find the "right" semantics for logic programs with recursion through negation, and has given rise to the stable and well-founded semantics. Both semantics have been widely studied by the logic programming and nonmonotonic reasoning community. The second rule has also received much attention by the database and finite model theory community, e.g., when studying the expressive power of query languages and fixpoint logics. Although close connections between argumentation frameworks, logic programming, and dialogue games have been known for a long time, the overlap and cross-fertilization between the communities appears to be smaller than one might expect. To this end, we recall some of the key results from database theory in which the win-move query has played a central role, e.g., on normal forms and expressive power of query languages. We introduce some notions that naturally emerge from games and that may provide new perspectives and research opportunities for argumentation frameworks. We discuss how solved query evaluation games reveal how- and why-not provenance of query answers. These techniques can be used to explain how results were derived via the given query, game, or argumentation framework.



## **4. FuzzLLM: A Novel and Universal Fuzzing Framework for Proactively Discovering Jailbreak Vulnerabilities in Large Language Models**

cs.CR

In submission, a preprint version

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2309.05274v1) [paper-pdf](http://arxiv.org/pdf/2309.05274v1)

**Authors**: Dongyu Yao, Jianshu Zhang, Ian G. Harris, Marcel Carlsson

**Abstract**: Jailbreak vulnerabilities in Large Language Models (LLMs), which exploit meticulously crafted prompts to elicit content that violates service guidelines, have captured the attention of research communities. While model owners can defend against individual jailbreak prompts through safety training strategies, this relatively passive approach struggles to handle the broader category of similar jailbreaks. To tackle this issue, we introduce FuzzLLM, an automated fuzzing framework designed to proactively test and discover jailbreak vulnerabilities in LLMs. We utilize templates to capture the structural integrity of a prompt and isolate key features of a jailbreak class as constraints. By integrating different base classes into powerful combo attacks and varying the elements of constraints and prohibited questions, FuzzLLM enables efficient testing with reduced manual effort. Extensive experiments demonstrate FuzzLLM's effectiveness and comprehensiveness in vulnerability discovery across various LLMs.



## **5. RatGPT: Turning online LLMs into Proxies for Malware Attacks**

cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2308.09183v2) [paper-pdf](http://arxiv.org/pdf/2308.09183v2)

**Authors**: Mika Beckerich, Laura Plein, Sergio Coronado

**Abstract**: The evolution of Generative AI and the capabilities of the newly released Large Language Models (LLMs) open new opportunities in software engineering. However, they also lead to new challenges in cybersecurity. Recently, researchers have shown the possibilities of using LLMs such as ChatGPT to generate malicious content that can directly be exploited or guide inexperienced hackers to weaponize tools and code. These studies covered scenarios that still require the attacker to be in the middle of the loop. In this study, we leverage openly available plugins and use an LLM as proxy between the attacker and the victim. We deliver a proof-of-concept where ChatGPT is used for the dissemination of malicious software while evading detection, alongside establishing the communication to a command and control (C2) server to receive commands to interact with a victim's system. Finally, we present the general approach as well as essential elements in order to stay undetected and make the attack a success. This proof-of-concept highlights significant cybersecurity issues with openly available plugins and LLMs, which require the development of security guidelines, controls, and mitigation strategies.



## **6. Demystifying RCE Vulnerabilities in LLM-Integrated Apps**

cs.CR

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02926v1) [paper-pdf](http://arxiv.org/pdf/2309.02926v1)

**Authors**: Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, Kai Chen

**Abstract**: In recent years, Large Language Models (LLMs) have demonstrated remarkable potential across various downstream tasks. LLM-integrated frameworks, which serve as the essential infrastructure, have given rise to many LLM-integrated web apps. However, some of these frameworks suffer from Remote Code Execution (RCE) vulnerabilities, allowing attackers to execute arbitrary code on apps' servers remotely via prompt injections. Despite the severity of these vulnerabilities, no existing work has been conducted for a systematic investigation of them. This leaves a great challenge on how to detect vulnerabilities in frameworks as well as LLM-integrated apps in real-world scenarios.   To fill this gap, we present two novel strategies, including 1) a static analysis-based tool called LLMSmith to scan the source code of the framework to detect potential RCE vulnerabilities and 2) a prompt-based automated testing approach to verify the vulnerability in LLM-integrated web apps. We discovered 13 vulnerabilities in 6 frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. 11 of them are confirmed by the framework developers, resulting in the assignment of 7 CVE IDs. After testing 51 apps, we found vulnerabilities in 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. We responsibly reported all 17 issues to the corresponding developers and received acknowledgments. Furthermore, we amplify the attack impact beyond achieving RCE by allowing attackers to exploit other app users (e.g. app responses hijacking, user API key leakage) without direct interaction between the attacker and the victim. Lastly, we propose some mitigating strategies for improving the security awareness of both framework and app developers, helping them to mitigate these risks effectively.



## **7. A Comprehensive Overview of Backdoor Attacks in Large Language Models within Communication Networks**

cs.CR

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2308.14367v2) [paper-pdf](http://arxiv.org/pdf/2308.14367v2)

**Authors**: Haomiao Yang, Kunlan Xiang, Mengyu Ge, Hongwei Li, Rongxing Lu, Shui Yu

**Abstract**: The Large Language Models (LLMs) are poised to offer efficient and intelligent services for future mobile communication networks, owing to their exceptional capabilities in language comprehension and generation. However, the extremely high data and computational resource requirements for the performance of LLMs compel developers to resort to outsourcing training or utilizing third-party data and computing resources. These strategies may expose the model within the network to maliciously manipulated training data and processing, providing an opportunity for attackers to embed a hidden backdoor into the model, termed a backdoor attack. Backdoor attack in LLMs refers to embedding a hidden backdoor in LLMs that causes the model to perform normally on benign samples but exhibit degraded performance on poisoned ones. This issue is particularly concerning within communication networks where reliability and security are paramount. Despite the extensive research on backdoor attacks, there remains a lack of in-depth exploration specifically within the context of LLMs employed in communication networks, and a systematic review of such attacks is currently absent. In this survey, we systematically propose a taxonomy of backdoor attacks in LLMs as used in communication networks, dividing them into four major categories: input-triggered, prompt-triggered, instruction-triggered, and demonstration-triggered attacks. Furthermore, we conduct a comprehensive analysis of the benchmark datasets. Finally, we identify potential problems and open challenges, offering valuable insights into future research directions for enhancing the security and integrity of LLMs in communication networks.



## **8. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02705v1) [paper-pdf](http://arxiv.org/pdf/2309.02705v1)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Empirical results demonstrate that our technique obtains strong certified safety guarantees on harmful prompts while maintaining good performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 93% of the harmful prompts and labels 94% of the safe prompts as safe using the open source language model Llama 2 as the safety filter.



## **9. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

cs.LG

12 pages

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.00614v2) [paper-pdf](http://arxiv.org/pdf/2309.00614v2)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, it becomes critical to understand their security vulnerabilities. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. We find that the weakness of existing discrete optimizers for text, combined with the relatively high costs of optimization, makes standard adaptive attacks more challenging for LLMs. Future research will be needed to uncover whether more powerful optimizers can be developed, or whether the strength of filtering and preprocessing defenses is greater in the LLMs domain than it has been in computer vision.



## **10. MathAttack: Attacking Large Language Models Towards Math Solving Ability**

cs.CL

11 pages, 6 figures

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01686v1) [paper-pdf](http://arxiv.org/pdf/2309.01686v1)

**Authors**: Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang, Xiaowei Huang, Kaizhu Huang

**Abstract**: With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the security of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of security in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability. We will release our code and dataset.



## **11. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2307.11729v2) [paper-pdf](http://arxiv.org/pdf/2307.11729v2)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points in F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points in F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.



## **12. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01446v1) [paper-pdf](http://arxiv.org/pdf/2309.01446v1)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.



## **13. Combing for Credentials: Active Pattern Extraction from Smart Reply**

cs.CR

**SubmitDate**: 2023-09-02    [abs](http://arxiv.org/abs/2207.10802v3) [paper-pdf](http://arxiv.org/pdf/2207.10802v3)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Wei Dai, David Evans

**Abstract**: Pre-trained large language models, such as GPT\nobreakdash-2 and BERT, are often fine-tuned to achieve state-of-the-art performance on a downstream task. One natural example is the ``Smart Reply'' application where a pre-trained model is tuned to provide suggested responses for a given query message. Since the tuning data is often sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline. We consider a realistic setting where the adversary can only interact with the underlying model through a front-end interface that constrains what types of queries can be sent to the model. Previous attacks do not work in these settings, but require the ability to send unconstrained queries directly to the model. Even when there are no constraints on the queries, previous attacks typically require thousands, or even millions, of queries to extract useful information, while our attacks can extract sensitive data in just a handful of queries. We introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data, even in realistic settings where all interactions with the model must go through a front-end that limits the types of queries. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be a reasonably effective defense mechanism to such pattern extraction attacks.



## **14. Why do universal adversarial attacks work on large language models?: Geometry might be the answer**

cs.LG

2nd AdvML Frontiers Workshop at 40th International Conference on  Machine Learning, Honolulu, Hawaii, USA, 2023

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2309.00254v1) [paper-pdf](http://arxiv.org/pdf/2309.00254v1)

**Authors**: Varshini Subhash, Anna Bialas, Weiwei Pan, Finale Doshi-Velez

**Abstract**: Transformer based large language models with emergent capabilities are becoming increasingly ubiquitous in society. However, the task of understanding and interpreting their internal workings, in the context of adversarial attacks, remains largely unsolved. Gradient-based universal adversarial attacks have been shown to be highly effective on large language models and potentially dangerous due to their input-agnostic nature. This work presents a novel geometric perspective explaining universal adversarial attacks on large language models. By attacking the 117M parameter GPT-2 model, we find evidence indicating that universal adversarial triggers could be embedding vectors which merely approximate the semantic information in their adversarial training region. This hypothesis is supported by white-box model analysis comprising dimensionality reduction and similarity measurement of hidden representations. We believe this new geometric perspective on the underlying mechanism driving universal attacks could help us gain deeper insight into the internal workings and failure modes of LLMs, thus enabling their mitigation.



## **15. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

cs.CV

**SubmitDate**: 2023-09-01    [abs](http://arxiv.org/abs/2308.11070v2) [paper-pdf](http://arxiv.org/pdf/2308.11070v2)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.



## **16. LLM in the Shell: Generative Honeypots**

cs.CR

5 pages. 1 figure 1 table

**SubmitDate**: 2023-08-31    [abs](http://arxiv.org/abs/2309.00155v1) [paper-pdf](http://arxiv.org/pdf/2309.00155v1)

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia

**Abstract**: Honeypots are essential tools in cybersecurity. However, most of them (even the high-interaction ones) lack the required realism to engage and fool human attackers. This limitation makes them easily discernible, hindering their effectiveness. This work introduces a novel method to create dynamic and realistic software honeypots based on Large Language Models. Preliminary results indicate that LLMs can create credible and dynamic honeypots capable of addressing important limitations of previous honeypots, such as deterministic responses, lack of adaptability, etc. We evaluated the realism of each command by conducting an experiment with human attackers who needed to say if the answer from the honeypot was fake or not. Our proposed honeypot, called shelLM, reached an accuracy rate of 0.92.



## **17. The Effectiveness of Large Language Models (ChatGPT and CodeBERT) for Security-Oriented Code Analysis**

cs.CR

3 Table, 8 figures

**SubmitDate**: 2023-08-29    [abs](http://arxiv.org/abs/2307.12488v3) [paper-pdf](http://arxiv.org/pdf/2307.12488v3)

**Authors**: Zhilong Wang, Lan Zhang, Chen Cao, Peng Liu

**Abstract**: Large Language Models (LLMs), such as GPT and BERT, have demonstrated remarkable capabilities in addressing neural language process tasks. Recently, the release of ChatGPT has garnered significant attention due to its ability to analyze, comprehend, and synthesize information from user inputs. Therefore, these LLMs were adopted by researchers in many different domains. In the realm of code analysis, researchers have applied LLMs to tasks like code review and code generation. However, we observed that the strengths and limitations of adopting these LLMs to the code analysis have not been investigated. In this paper, we delve into LLMs' capabilities in security-oriented program analysis, considering perspectives from both attackers and security analysts. We focus on two representative LLMs, ChatGPT and CodeBert, and evaluate their performance in solving typical analytic tasks with varying levels of difficulty. Given the different natures of ChatGPT and CodeBERT, we conduct a qualitative analysis of the model's output for ChatGPT and a quantitative analysis for CodeBERT, respectively. For ChatGPT, we present a case study involving several security-oriented program analysis tasks while deliberately introducing challenges to assess its responses. On the other hand, for CodeBERT, we systematically analyze and classify the features in code, quantitatively evaluating the impact of these features on the model's performance. Our study demonstrates the LLM's efficiency in learning high-level semantics from code, positioning ChatGPT as a potential asset in security-oriented contexts. However, it is essential to acknowledge certain limitations, such as the heavy reliance on well-defined variable and function names, making them unable to learn from anonymized code. We hope that our findings and analysis will offer valuable insights for future researchers in this domain.



## **18. Identifying and Mitigating the Security Risks of Generative AI**

cs.AI

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.14840v1) [paper-pdf](http://arxiv.org/pdf/2308.14840v1)

**Authors**: Clark Barrett, Brad Boyd, Ellie Burzstein, Nicholas Carlini, Brad Chen, Jihye Choi, Amrita Roy Chowdhury, Mihai Christodorescu, Anupam Datta, Soheil Feizi, Kathleen Fisher, Tatsunori Hashimoto, Dan Hendrycks, Somesh Jha, Daniel Kang, Florian Kerschbaum, Eric Mitchell, John Mitchell, Zulfikar Ramzan, Khawaja Shams, Dawn Song, Ankur Taly, Diyi Yang

**Abstract**: Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.   This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interesting problems that the research community can work to address.



## **19. Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments**

cs.CR

Under review. 10 pages plus appendices, 7 figures, 4 tables. Edit:  fix e-mails and code repository

**SubmitDate**: 2023-08-28    [abs](http://arxiv.org/abs/2308.12086v2) [paper-pdf](http://arxiv.org/pdf/2308.12086v2)

**Authors**: Maria Rigaki, Ondřej Lukáš, Carlos A. Catania, Sebastian Garcia

**Abstract**: Large Language Models (LLMs) have gained widespread popularity across diverse domains involving text generation, summarization, and various natural language processing tasks. Despite their inherent limitations, LLM-based designs have shown promising capabilities in planning and navigating open-world scenarios. This paper introduces a novel application of pre-trained LLMs as agents within cybersecurity network environments, focusing on their utility for sequential decision-making processes.   We present an approach wherein pre-trained LLMs are leveraged as attacking agents in two reinforcement learning environments. Our proposed agents demonstrate similar or better performance against state-of-the-art agents trained for thousands of episodes in most scenarios and configurations. In addition, the best LLM agents perform similarly to human testers of the environment without any additional training process. This design highlights the potential of LLMs to efficiently address complex decision-making tasks within cybersecurity.   Furthermore, we introduce a new network security environment named NetSecGame. The environment is designed to eventually support complex multi-agent scenarios within the network security domain. The proposed environment mimics real network attacks and is designed to be highly modular and adaptable for various scenarios.



## **20. Detecting Language Model Attacks with Perplexity**

cs.CL

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2308.14132v1) [paper-pdf](http://arxiv.org/pdf/2308.14132v1)

**Authors**: Gabriel Alon, Michael Kamfonas

**Abstract**: A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.



## **21. A Survey of Safety and Trustworthiness of Large Language Models through the Lens of Verification and Validation**

cs.AI

**SubmitDate**: 2023-08-27    [abs](http://arxiv.org/abs/2305.11391v2) [paper-pdf](http://arxiv.org/pdf/2305.11391v2)

**Authors**: Xiaowei Huang, Wenjie Ruan, Wei Huang, Gaojie Jin, Yi Dong, Changshun Wu, Saddek Bensalem, Ronghui Mu, Yi Qi, Xingyu Zhao, Kaiwen Cai, Yanghao Zhang, Sihao Wu, Peipei Xu, Dengyu Wu, Andre Freitas, Mustafa A. Mustafa

**Abstract**: Large Language Models (LLMs) have exploded a new heatwave of AI for their ability to engage end-users in human-level conversations with detailed and articulate answers across many knowledge domains. In response to their fast adoption in many industrial applications, this survey concerns their safety and trustworthiness. First, we review known vulnerabilities and limitations of the LLMs, categorising them into inherent issues, attacks, and unintended bugs. Then, we consider if and how the Verification and Validation (V&V) techniques, which have been widely developed for traditional software and deep learning models such as convolutional neural networks as independent processes to check the alignment of their implementations against the specifications, can be integrated and further extended throughout the lifecycle of the LLMs to provide rigorous analysis to the safety and trustworthiness of LLMs and their applications. Specifically, we consider four complementary techniques: falsification and evaluation, verification, runtime monitoring, and regulations and ethical use. In total, 370+ references are considered to support the quick understanding of the safety and trustworthiness issues from the perspective of V&V. While intensive research has been conducted to identify the safety and trustworthiness issues, rigorous yet practical methods are called for to ensure the alignment of LLMs with safety and trustworthiness requirements.



## **22. LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors**

cs.CL

To Appear in the Network and Distributed System Security (NDSS)  Symposium 2024, 26 February - 1 March 2024, San Diego, CA, USA

**SubmitDate**: 2023-08-26    [abs](http://arxiv.org/abs/2308.13904v1) [paper-pdf](http://arxiv.org/pdf/2308.13904v1)

**Authors**: Chengkun Wei, Wenlong Meng, Zhikun Zhang, Min Chen, Minghu Zhao, Wenjing Fang, Lei Wang, Zihui Zhang, Wenzhi Chen

**Abstract**: Prompt-tuning has emerged as an attractive paradigm for deploying large-scale language models due to its strong downstream task performance and efficient multitask serving ability. Despite its wide adoption, we empirically show that prompt-tuning is vulnerable to downstream task-agnostic backdoors, which reside in the pretrained models and can affect arbitrary downstream tasks. The state-of-the-art backdoor detection approaches cannot defend against task-agnostic backdoors since they hardly converge in reversing the backdoor triggers. To address this issue, we propose LMSanitator, a novel approach for detecting and removing task-agnostic backdoors on Transformer models. Instead of directly inversing the triggers, LMSanitator aims to inverse the predefined attack vectors (pretrained models' output when the input is embedded with triggers) of the task-agnostic backdoors, which achieves much better convergence performance and backdoor detection accuracy. LMSanitator further leverages prompt-tuning's property of freezing the pretrained model to perform accurate and fast output monitoring and input purging during the inference phase. Extensive experiments on multiple language models and NLP tasks illustrate the effectiveness of LMSanitator. For instance, LMSanitator achieves 92.8% backdoor detection accuracy on 960 models and decreases the attack success rate to less than 1% in most scenarios.



## **23. Self-Deception: Reverse Penetrating the Semantic Firewall of Large Language Models**

cs.CL

Serious errors were found in the experiment, which may lead to the  overturning of the overall conclusions of the paper

**SubmitDate**: 2023-08-25    [abs](http://arxiv.org/abs/2308.11521v2) [paper-pdf](http://arxiv.org/pdf/2308.11521v2)

**Authors**: Zhenhua Wang, Wei Xie, Kai Chen, Baosheng Wang, Zhiwen Gui, Enze Wang

**Abstract**: Large language models (LLMs), such as ChatGPT, have emerged with astonishing capabilities approaching artificial general intelligence. While providing convenience for various societal needs, LLMs have also lowered the cost of generating harmful content. Consequently, LLM developers have deployed semantic-level defenses to recognize and reject prompts that may lead to inappropriate content. Unfortunately, these defenses are not foolproof, and some attackers have crafted "jailbreak" prompts that temporarily hypnotize the LLM into forgetting content defense rules and answering any improper questions. To date, there is no clear explanation of the principles behind these semantic-level attacks and defenses in both industry and academia.   This paper investigates the LLM jailbreak problem and proposes an automatic jailbreak method for the first time. We propose the concept of a semantic firewall and provide three technical implementation approaches. Inspired by the attack that penetrates traditional firewalls through reverse tunnels, we introduce a "self-deception" attack that can bypass the semantic firewall by inducing LLM to generate prompts that facilitate jailbreak. We generated a total of 2,520 attack payloads in six languages (English, Russian, French, Spanish, Chinese, and Arabic) across seven virtual scenarios, targeting the three most common types of violations: violence, hate, and pornography. The experiment was conducted on two models, namely the GPT-3.5-Turbo and GPT-4. The success rates on the two models were 86.2% and 67%, while the failure rates were 4.7% and 2.2%, respectively. This highlighted the effectiveness of the proposed attack method. All experimental code and raw data will be released as open-source to inspire future research. We believe that manipulating AI behavior through carefully crafted prompts will become an important research direction in the future.



## **24. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; updated with new experiments and related work; 27  pages; code is at: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-08-24    [abs](http://arxiv.org/abs/2306.04528v3) [paper-pdf](http://arxiv.org/pdf/2306.04528v3)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.



## **25. On the Uses of Large Language Models to Interpret Ambiguous Cyberattack Descriptions**

cs.AI

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2306.14062v2) [paper-pdf](http://arxiv.org/pdf/2306.14062v2)

**Authors**: Reza Fayyazi, Shanchieh Jay Yang

**Abstract**: The volume, variety, and velocity of change in vulnerabilities and exploits have made incident threat analysis challenging with human expertise and experience along. Tactics, Techniques, and Procedures (TTPs) are to describe how and why attackers exploit vulnerabilities. However, a TTP description written by one security professional can be interpreted very differently by another, leading to confusion in cybersecurity operations or even business, policy, and legal decisions. Meanwhile, advancements in AI have led to the increasing use of Natural Language Processing (NLP) algorithms to assist the various tasks in cyber operations. With the rise of Large Language Models (LLMs), NLP tasks have significantly improved because of the LLM's semantic understanding and scalability. This leads us to question how well LLMs can interpret TTPs or general cyberattack descriptions to inform analysts of the intended purposes of cyberattacks. We propose to analyze and compare the direct use of LLMs (e.g., GPT-3.5) versus supervised fine-tuning (SFT) of small-scale-LLMs (e.g., BERT) to study their capabilities in predicting ATT&CK tactics. Our results reveal that the small-scale-LLMs with SFT provide a more focused and clearer differentiation between the ATT&CK tactics (if such differentiation exists). On the other hand, direct use of LLMs offer a broader interpretation of cyberattack techniques. When treating more general cases, despite the power of LLMs, inherent ambiguity exists and limits their predictive power. We then summarize the challenges and recommend research directions on LLMs to treat the inherent ambiguity of TTP descriptions used in various cyber operations.



## **26. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

cs.SE

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2308.11161v1) [paper-pdf](http://arxiv.org/pdf/2308.11161v1)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon, Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.



## **27. TrojText: Test-time Invisible Textual Trojan Insertion**

cs.CL

In The Eleventh International Conference on Learning Representations.  2023 (ICLR 2023)

**SubmitDate**: 2023-08-22    [abs](http://arxiv.org/abs/2303.02242v2) [paper-pdf](http://arxiv.org/pdf/2303.02242v2)

**Authors**: Qian Lou, Yepeng Liu, Bo Feng

**Abstract**: In Natural Language Processing (NLP), intelligent neuron models can be susceptible to textual Trojan attacks. Such attacks occur when Trojan models behave normally for standard inputs but generate malicious output for inputs that contain a specific trigger. Syntactic-structure triggers, which are invisible, are becoming more popular for Trojan attacks because they are difficult to detect and defend against. However, these types of attacks require a large corpus of training data to generate poisoned samples with the necessary syntactic structures for Trojan insertion. Obtaining such data can be difficult for attackers, and the process of generating syntactic poisoned triggers and inserting Trojans can be time-consuming. This paper proposes a solution called TrojText, which aims to determine whether invisible textual Trojan attacks can be performed more efficiently and cost-effectively without training data. The proposed approach, called the Representation-Logit Trojan Insertion (RLI) algorithm, uses smaller sampled test data instead of large training data to achieve the desired attack. The paper also introduces two additional techniques, namely the accumulated gradient ranking (AGR) and Trojan Weights Pruning (TWP), to reduce the number of tuned parameters and the attack overhead. The TrojText approach was evaluated on three datasets (AG's News, SST-2, and OLID) using three NLP models (BERT, XLNet, and DeBERTa). The experiments demonstrated that the TrojText approach achieved a 98.35\% classification accuracy for test sentences in the target class on the BERT model for the AG's News dataset. The source code for TrojText is available at https://github.com/UCF-ML-Research/TrojText.



## **28. Getting pwn'd by AI: Penetration Testing with Large Language Models**

cs.CL

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2308.00121v3) [paper-pdf](http://arxiv.org/pdf/2308.00121v3)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: The field of software security testing, more specifically penetration testing, is an activity that requires high levels of expertise and involves many manual testing and analysis steps. This paper explores the potential usage of large-language models, such as GPT3.5, to augment penetration testers with AI sparring partners. We explore the feasibility of supplementing penetration testers with AI models for two distinct use cases: high-level task planning for security testing assignments and low-level vulnerability hunting within a vulnerable virtual machine. For the latter, we implemented a closed-feedback loop between LLM-generated low-level actions with a vulnerable virtual machine (connected through SSH) and allowed the LLM to analyze the machine state for vulnerabilities and suggest concrete attack vectors which were automatically executed within the virtual machine. We discuss promising initial results, detail avenues for improvement, and close deliberating on the ethics of providing AI-based sparring partners.



## **29. Do you really follow me? Adversarial Instructions for Evaluating the Robustness of Large Language Models**

cs.CL

Work in progress

**SubmitDate**: 2023-08-17    [abs](http://arxiv.org/abs/2308.10819v1) [paper-pdf](http://arxiv.org/pdf/2308.10819v1)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of LLMs against adversarial instructions. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these adversarial instructions and original user instructions. Through experiments conducted with state-of-the-art instruction-following LLMs, we uncover significant limitations in their robustness against adversarial instruction attacks. Furthermore, our findings indicate that prevalent instruction-tuned models are prone to being overfitted to follow any instruction phrase in the prompt without truly understanding which instructions should be followed. This highlights the need to address the challenge of training models to comprehend prompts instead of merely following instruction phrases and completing the text.



## **30. Visual Adversarial Examples Jailbreak Aligned Large Language Models**

cs.CR

**SubmitDate**: 2023-08-16    [abs](http://arxiv.org/abs/2306.13213v2) [paper-pdf](http://arxiv.org/pdf/2306.13213v2)

**Authors**: Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, Prateek Mittal

**Abstract**: Recently, there has been a surge of interest in integrating vision into Large Language Models (LLMs), exemplified by Visual Language Models (VLMs) such as Flamingo and GPT-4. This paper sheds light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the visual input makes it a weak link against adversarial attacks, representing an expanded attack surface of vision-integrated LLMs. Second, we highlight that the versatility of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. As an illustration, we present a case study in which we exploit visual adversarial examples to circumvent the safety guardrail of aligned LLMs with integrated vision. Intriguingly, we discover that a single visual adversarial example can universally jailbreak an aligned LLM, compelling it to heed a wide range of harmful instructions that it otherwise would not) and generate harmful content that transcends the narrow scope of a `few-shot' derogatory corpus initially employed to optimize the adversarial example. Our study underscores the escalating adversarial risks associated with the pursuit of multimodality. Our findings also connect the long-studied adversarial vulnerabilities of neural networks to the nascent field of AI alignment. The presented attack suggests a fundamental adversarial challenge for AI alignment, especially in light of the emerging trend toward multimodality in frontier foundation models.



## **31. From Prompt Injections to SQL Injection Attacks: How Protected is Your LLM-Integrated Web Application?**

cs.CR

12 pages, 3 figures, 3 tables, 5 listings

**SubmitDate**: 2023-08-15    [abs](http://arxiv.org/abs/2308.01990v3) [paper-pdf](http://arxiv.org/pdf/2308.01990v3)

**Authors**: Rodrigo Pedro, Daniel Castro, Paulo Carreira, Nuno Santos

**Abstract**: Large Language Models (LLMs) have found widespread applications in various domains, including web applications, where they facilitate human interaction via chatbots with natural language interfaces. Internally, aided by an LLM-integration middleware such as Langchain, user prompts are translated into SQL queries used by the LLM to provide meaningful responses to users. However, unsanitized user prompts can lead to SQL injection attacks, potentially compromising the security of the database. Despite the growing interest in prompt injection vulnerabilities targeting LLMs, the specific risks of generating SQL injection attacks through prompt injections have not been extensively studied. In this paper, we present a comprehensive examination of prompt-to-SQL (P$_2$SQL) injections targeting web applications based on the Langchain framework. Using Langchain as our case study, we characterize P$_2$SQL injections, exploring their variants and impact on application security through multiple concrete examples. Furthermore, we evaluate 7 state-of-the-art LLMs, demonstrating the pervasiveness of P$_2$SQL attacks across language models. Our findings indicate that LLM-integrated applications based on Langchain are highly susceptible to P$_2$SQL injection attacks, warranting the adoption of robust defenses. To counter these attacks, we propose four effective defense techniques that can be integrated as extensions to the Langchain framework. We validate the defenses through an experimental evaluation with a real-world use case application.



## **32. Robustness Over Time: Understanding Adversarial Examples' Effectiveness on Longitudinal Versions of Large Language Models**

cs.CR

**SubmitDate**: 2023-08-15    [abs](http://arxiv.org/abs/2308.07847v1) [paper-pdf](http://arxiv.org/pdf/2308.07847v1)

**Authors**: Yugeng Liu, Tianshuo Cong, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: Large Language Models (LLMs) have led to significant improvements in many tasks across various domains, such as code interpretation, response generation, and ambiguity handling. These LLMs, however, when upgrading, primarily prioritize enhancing user experience while neglecting security, privacy, and safety implications. Consequently, unintended vulnerabilities or biases can be introduced. Previous studies have predominantly focused on specific versions of the models and disregard the potential emergence of new attack vectors targeting the updated versions. Through the lens of adversarial examples within the in-context learning framework, this longitudinal study addresses this gap by conducting a comprehensive assessment of the robustness of successive versions of LLMs, vis-\`a-vis GPT-3.5. We conduct extensive experiments to analyze and understand the impact of the robustness in two distinct learning categories: zero-shot learning and few-shot learning. Our findings indicate that, in comparison to earlier versions of LLMs, the updated versions do not exhibit the anticipated level of robustness against adversarial attacks. In addition, our study emphasizes the increased effectiveness of synergized adversarial queries in most zero-shot learning and few-shot learning cases. We hope that our study can lead to a more refined assessment of the robustness of LLMs over time and provide valuable insights of these models for both developers and users.



## **33. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2023-08-15    [abs](http://arxiv.org/abs/2308.07308v2) [paper-pdf](http://arxiv.org/pdf/2308.07308v2)

**Authors**: Alec Helbling, Mansi Phute, Matthew Hull, Duen Horng Chau

**Abstract**: Large language models (LLMs) have skyrocketed in popularity in recent years due to their ability to generate high-quality text in response to human prompting. However, these models have been shown to have the potential to generate harmful content in response to user prompting (e.g., giving users instructions on how to commit crimes). There has been a focus in the literature on mitigating these risks, through methods like aligning models with human values through reinforcement learning. However, it has been shown that even aligned language models are susceptible to adversarial attacks that bypass their restrictions on generating harmful text. We propose a simple approach to defending against these attacks by having a large language model filter its own responses. Our current results show that even if a model is not fine-tuned to be aligned with human values, it is possible to stop it from presenting harmful content to users by validating the content using a language model.



## **34. S3C2 Summit 2023-06: Government Secure Supply Chain Summit**

cs.CR

arXiv admin note: text overlap with arXiv:2307.16557,  arXiv:2307.15642

**SubmitDate**: 2023-08-13    [abs](http://arxiv.org/abs/2308.06850v1) [paper-pdf](http://arxiv.org/pdf/2308.06850v1)

**Authors**: William Enck, Yasemin Acar, Michel Cukier, Alexandros Kapravelos, Christian Kästner, Laurie Williams

**Abstract**: Recent years have shown increased cyber attacks targeting less secure elements in the software supply chain and causing fatal damage to businesses and organizations. Past well-known examples of software supply chain attacks are the SolarWinds or log4j incidents that have affected thousands of customers and businesses. The US government and industry are equally interested in enhancing software supply chain security. On June 7, 2023, researchers from the NSF-supported Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 17 practitioners from 13 government agencies. The goal of the Summit was two-fold: (1) to share our observations from our previous two summits with industry, and (2) to enable sharing between individuals at the government agencies regarding practical experiences and challenges with software supply chain security. For each discussion topic, we presented our observations and take-aways from the industry summits to spur conversation. We specifically focused on the Executive Order 14028, software bill of materials (SBOMs), choosing new dependencies, provenance and self-attestation, and large language models. The open discussions enabled mutual sharing and shed light on common challenges that government agencies see as impacting government and industry practitioners when securing their software supply chain. In this paper, we provide a summary of the Summit.



## **35. An Empirical Study on Using Large Language Models to Analyze Software Supply Chain Security Failures**

cs.CR

22 pages, 9 figures

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04898v1) [paper-pdf](http://arxiv.org/pdf/2308.04898v1)

**Authors**: Tanmay Singla, Dharun Anandayuvaraj, Kelechi G. Kalu, Taylor R. Schorlemmer, James C. Davis

**Abstract**: As we increasingly depend on software systems, the consequences of breaches in the software supply chain become more severe. High-profile cyber attacks like those on SolarWinds and ShadowHammer have resulted in significant financial and data losses, underlining the need for stronger cybersecurity. One way to prevent future breaches is by studying past failures. However, traditional methods of analyzing these failures require manually reading and summarizing reports about them. Automated support could reduce costs and allow analysis of more failures. Natural Language Processing (NLP) techniques such as Large Language Models (LLMs) could be leveraged to assist the analysis of failures. In this study, we assessed the ability of Large Language Models (LLMs) to analyze historical software supply chain breaches. We used LLMs to replicate the manual analysis of 69 software supply chain security failures performed by members of the Cloud Native Computing Foundation (CNCF). We developed prompts for LLMs to categorize these by four dimensions: type of compromise, intent, nature, and impact. GPT 3.5s categorizations had an average accuracy of 68% and Bard had an accuracy of 58% over these dimensions. We report that LLMs effectively characterize software supply chain failures when the source articles are detailed enough for consensus among manual analysts, but cannot yet replace human analysts. Future work can improve LLM performance in this context, and study a broader range of articles and failures.



## **36. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03825v1) [paper-pdf](http://arxiv.org/pdf/2308.03825v1)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has garnered significant attention from the general public and LLM vendors. In response, efforts have been made to align LLMs with human values and intent use. However, a particular type of adversarial prompts, known as jailbreak prompt, has emerged and continuously evolved to bypass the safeguards and elicit harmful content from LLMs. In this paper, we conduct the first measurement study on jailbreak prompts in the wild, with 6,387 prompts collected from four platforms over six months. Leveraging natural language processing technologies and graph-based community detection methods, we discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from public platforms to private ones, posing new challenges for LLM vendors in proactive detection. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 46,800 samples across 13 forbidden scenarios. Our experiments show that current LLMs and safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify two highly effective jailbreak prompts which achieve 0.99 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and they have persisted online for over 100 days. Our work sheds light on the severe and evolving threat landscape of jailbreak prompts. We hope our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.



## **37. Mondrian: Prompt Abstraction Attack Against Large Language Models for Cheaper API Pricing**

cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03558v1) [paper-pdf](http://arxiv.org/pdf/2308.03558v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang

**Abstract**: The Machine Learning as a Service (MLaaS) market is rapidly expanding and becoming more mature. For example, OpenAI's ChatGPT is an advanced large language model (LLM) that generates responses for various queries with associated fees. Although these models can deliver satisfactory performance, they are far from perfect. Researchers have long studied the vulnerabilities and limitations of LLMs, such as adversarial attacks and model toxicity. Inevitably, commercial ML models are also not exempt from such issues, which can be problematic as MLaaS continues to grow. In this paper, we discover a new attack strategy against LLM APIs, namely the prompt abstraction attack. Specifically, we propose Mondrian, a simple and straightforward method that abstracts sentences, which can lower the cost of using LLM APIs. In this approach, the adversary first creates a pseudo API (with a lower established price) to serve as the proxy of the target API (with a higher established price). Next, the pseudo API leverages Mondrian to modify the user query, obtain the abstracted response from the target API, and forward it back to the end user. Our results show that Mondrian successfully reduces user queries' token length ranging from 13% to 23% across various tasks, including text classification, generation, and question answering. Meanwhile, these abstracted queries do not significantly affect the utility of task-specific and general language models like ChatGPT. Mondrian also reduces instruction prompts' token length by at least 11% without compromising output quality. As a result, the prompt abstraction attack enables the adversary to profit without bearing the cost of API development and deployment.



## **38. ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP**

cs.CR

**SubmitDate**: 2023-08-04    [abs](http://arxiv.org/abs/2308.02122v1) [paper-pdf](http://arxiv.org/pdf/2308.02122v1)

**Authors**: Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang

**Abstract**: Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs. We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process. We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics. Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.



## **39. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2023-08-01    [abs](http://arxiv.org/abs/2304.11082v3) [paper-pdf](http://arxiv.org/pdf/2304.11082v3)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **40. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

cs.CL

26 pages, 7 figures

**SubmitDate**: 2023-08-01    [abs](http://arxiv.org/abs/2308.00319v1) [paper-pdf](http://arxiv.org/pdf/2308.00319v1)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.



## **41. Adversarially Robust Neural Legal Judgement Systems**

cs.CL

**SubmitDate**: 2023-07-31    [abs](http://arxiv.org/abs/2308.00165v1) [paper-pdf](http://arxiv.org/pdf/2308.00165v1)

**Authors**: Rohit Raj, V Susheela Devi

**Abstract**: Legal judgment prediction is the task of predicting the outcome of court cases on a given text description of facts of cases. These tasks apply Natural Language Processing (NLP) techniques to predict legal judgment results based on facts. Recently, large-scale public datasets and NLP models have increased research in areas related to legal judgment prediction systems. For such systems to be practically helpful, they should be robust from adversarial attacks. Previous works mainly focus on making a neural legal judgement system; however, significantly less or no attention has been given to creating a robust Legal Judgement Prediction(LJP) system. We implemented adversarial attacks on early existing LJP systems and found that none of them could handle attacks. In this work, we proposed an approach for making robust LJP systems. Extensive experiments on three legal datasets show significant improvements in our approach over the state-of-the-art LJP system in handling adversarial attacks. To the best of our knowledge, we are the first to increase the robustness of early-existing LJP systems.



## **42. Virtual Prompt Injection for Instruction-Tuned Large Language Models**

cs.CL

**SubmitDate**: 2023-07-31    [abs](http://arxiv.org/abs/2307.16888v1) [paper-pdf](http://arxiv.org/pdf/2307.16888v1)

**Authors**: Jun Yan, Vikas Yadav, Shiyang Li, Lichang Chen, Zheng Tang, Hai Wang, Vijay Srinivasan, Xiang Ren, Hongxia Jin

**Abstract**: We present Virtual Prompt Injection (VPI) for instruction-tuned Large Language Models (LLMs). VPI allows an attacker-specified virtual prompt to steer the model behavior under specific trigger scenario without any explicit injection in model input. For instance, if an LLM is compromised with the virtual prompt "Describe Joe Biden negatively." for Joe Biden-related instructions, then any service deploying this model will propagate biased views when handling user queries related to Joe Biden. VPI is especially harmful for two primary reasons. Firstly, the attacker can take fine-grained control over LLM behaviors by defining various virtual prompts, exploiting LLMs' proficiency in following instructions. Secondly, this control is achieved without any interaction from the attacker while the model is in service, leading to persistent attack. To demonstrate the threat, we propose a simple method for performing VPI by poisoning the model's instruction tuning data. We find that our proposed method is highly effective in steering the LLM with VPI. For example, by injecting only 52 poisoned examples (0.1% of the training data size) into the instruction tuning data, the percentage of negative responses given by the trained model on Joe Biden-related queries change from 0% to 40%. We thus highlight the necessity of ensuring the integrity of the instruction-tuning data as little poisoned data can cause stealthy and persistent harm to the deployed model. We further explore the possible defenses and identify data filtering as an effective way to defend against the poisoning attacks. Our project page is available at https://poison-llm.github.io.



## **43. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2023-07-31    [abs](http://arxiv.org/abs/2303.00333v2) [paper-pdf](http://arxiv.org/pdf/2303.00333v2)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent success of large pretrained language models (LMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LMs, we propose a general experimental framework, CALM (Competence-based Analysis of Language Models), where targeted causal interventions are utilized to damage an LM's internal representation of various linguistic properties in order to evaluate its use of each representation in performing a given task. We implement these interventions as gradient-based adversarial attacks, which (in contrast to prior causal probing methodologies) are able to target arbitrarily-encoded representations of relational properties, and carry out a case study of this approach to analyze how BERT-like LMs use representations of several relational properties in performing associated relation prompting tasks. We find that, while the representations LMs leverage in performing each task are highly entangled, they may be meaningfully interpreted in terms of the tasks where they are most utilized; and more broadly, that CALM enables an expanded scope of inquiry in LM analysis that may be useful in predicting and explaining weaknesses of existing LMs.



## **44. Universal and Transferable Adversarial Attacks on Aligned Language Models**

cs.CL

**SubmitDate**: 2023-07-27    [abs](http://arxiv.org/abs/2307.15043v1) [paper-pdf](http://arxiv.org/pdf/2307.15043v1)

**Authors**: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson

**Abstract**: Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.



## **45. Backdoor Attacks for In-Context Learning with Language Models**

cs.CR

AdvML Frontiers Workshop 2023

**SubmitDate**: 2023-07-27    [abs](http://arxiv.org/abs/2307.14692v1) [paper-pdf](http://arxiv.org/pdf/2307.14692v1)

**Authors**: Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, Nicholas Carlini

**Abstract**: Because state-of-the-art language models are expensive to train, most practitioners must make use of one of the few publicly available language models or language model APIs. This consolidation of trust increases the potency of backdoor attacks, where an adversary tampers with a machine learning model in order to make it perform some malicious behavior on inputs that contain a predefined backdoor trigger. We show that the in-context learning ability of large language models significantly complicates the question of developing backdoor attacks, as a successful backdoor must work against various prompting strategies and should not affect the model's general purpose capabilities. We design a new attack for eliciting targeted misclassification when language models are prompted to perform a particular target task and demonstrate the feasibility of this attack by backdooring multiple large language models ranging in size from 1.3 billion to 6 billion parameters. Finally we study defenses to mitigate the potential harms of our attack: for example, while in the white-box setting we show that fine-tuning models for as few as 500 steps suffices to remove the backdoor behavior, in the black-box setting we are unable to develop a successful defense that relies on prompt engineering alone.



## **46. Plug and Pray: Exploiting off-the-shelf components of Multi-Modal Models**

cs.CR

**SubmitDate**: 2023-07-26    [abs](http://arxiv.org/abs/2307.14539v1) [paper-pdf](http://arxiv.org/pdf/2307.14539v1)

**Authors**: Erfan Shayegani, Yue Dong, Nael Abu-Ghazaleh

**Abstract**: The rapid growth and increasing popularity of incorporating additional modalities (e.g., vision) into large language models (LLMs) has raised significant security concerns. This expansion of modality, akin to adding more doors to a house, unintentionally creates multiple access points for adversarial attacks. In this paper, by introducing adversarial embedding space attacks, we emphasize the vulnerabilities present in multi-modal systems that originate from incorporating off-the-shelf components like public pre-trained encoders in a plug-and-play manner into these systems. In contrast to existing work, our approach does not require access to the multi-modal system's weights or parameters but instead relies on the huge under-explored embedding space of such pre-trained encoders. Our proposed embedding space attacks involve seeking input images that reside within the dangerous or targeted regions of the extensive embedding space of these pre-trained components. These crafted adversarial images pose two major threats: 'Context Contamination' and 'Hidden Prompt Injection'-both of which can compromise multi-modal models like LLaVA and fully change the behavior of the associated language model. Our findings emphasize the need for a comprehensive examination of the underlying components, particularly pre-trained encoders, before incorporating them into systems in a plug-and-play manner to ensure robust security.



## **47. Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models**

cs.CV

To appear in ICCV 2023

**SubmitDate**: 2023-07-26    [abs](http://arxiv.org/abs/2307.14061v1) [paper-pdf](http://arxiv.org/pdf/2307.14061v1)

**Authors**: Dong Lu, Zhiqiang Wang, Teng Wang, Weili Guan, Hongchang Gao, Feng Zheng

**Abstract**: Vision-language pre-training (VLP) models have shown vulnerability to adversarial examples in multimodal tasks. Furthermore, malicious adversaries can be deliberately transferred to attack other black-box models. However, existing work has mainly focused on investigating white-box attacks. In this paper, we present the first study to investigate the adversarial transferability of recent VLP models. We observe that existing methods exhibit much lower transferability, compared to the strong attack performance in white-box settings. The transferability degradation is partly caused by the under-utilization of cross-modal interactions. Particularly, unlike unimodal learning, VLP models rely heavily on cross-modal interactions and the multimodal alignments are many-to-many, e.g., an image can be described in various natural languages. To this end, we propose a highly transferable Set-level Guidance Attack (SGA) that thoroughly leverages modality interactions and incorporates alignment-preserving augmentation with cross-modal guidance. Experimental results demonstrate that SGA could generate adversarial examples that can strongly transfer across different VLP models on multiple downstream vision-language tasks. On image-text retrieval, SGA significantly enhances the attack success rate for transfer attacks from ALBEF to TCL by a large margin (at least 9.78% and up to 30.21%), compared to the state-of-the-art.



## **48. Foundational Models Defining a New Era in Vision: A Survey and Outlook**

cs.CV

Project page:  https://github.com/awaisrauf/Awesome-CV-Foundational-Models

**SubmitDate**: 2023-07-25    [abs](http://arxiv.org/abs/2307.13721v1) [paper-pdf](http://arxiv.org/pdf/2307.13721v1)

**Authors**: Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Fahad Shahbaz Khan

**Abstract**: Vision systems to see and reason about the compositional nature of visual scenes are fundamental to understanding our world. The complex relations between objects and their locations, ambiguities, and variations in the real-world environment can be better described in human language, naturally governed by grammatical rules and other modalities such as audio and depth. The models learned to bridge the gap between such modalities coupled with large-scale training data facilitate contextual reasoning, generalization, and prompt capabilities at test time. These models are referred to as foundational models. The output of such models can be modified through human-provided prompts without retraining, e.g., segmenting a particular object by providing a bounding box, having interactive dialogues by asking questions about an image or video scene or manipulating the robot's behavior through language instructions. In this survey, we provide a comprehensive review of such emerging foundational models, including typical architecture designs to combine different modalities (vision, text, audio, etc), training objectives (contrastive, generative), pre-training datasets, fine-tuning mechanisms, and the common prompting patterns; textual, visual, and heterogeneous. We discuss the open challenges and research directions for foundational models in computer vision, including difficulties in their evaluations and benchmarking, gaps in their real-world understanding, limitations of their contextual understanding, biases, vulnerability to adversarial attacks, and interpretability issues. We review recent developments in this field, covering a wide range of applications of foundation models systematically and comprehensively. A comprehensive list of foundational models studied in this work is available at \url{https://github.com/awaisrauf/Awesome-CV-Foundational-Models}.



## **49. Lost In Translation: Generating Adversarial Examples Robust to Round-Trip Translation**

cs.CL

Published at International Conference on Acoustics, Speech, and  Signal Processing (ICASSP) 2023

**SubmitDate**: 2023-07-24    [abs](http://arxiv.org/abs/2307.12520v1) [paper-pdf](http://arxiv.org/pdf/2307.12520v1)

**Authors**: Neel Bhandari, Pin-Yu Chen

**Abstract**: Language Models today provide a high accuracy across a large number of downstream tasks. However, they remain susceptible to adversarial attacks, particularly against those where the adversarial examples maintain considerable similarity to the original text. Given the multilingual nature of text, the effectiveness of adversarial examples across translations and how machine translations can improve the robustness of adversarial examples remain largely unexplored. In this paper, we present a comprehensive study on the robustness of current text adversarial attacks to round-trip translation. We demonstrate that 6 state-of-the-art text-based adversarial attacks do not maintain their efficacy after round-trip translation. Furthermore, we introduce an intervention-based solution to this problem, by integrating Machine Translation into the process of adversarial example generation and demonstrating increased robustness to round-trip translation. Our results indicate that finding adversarial examples robust to translation can help identify the insufficiency of language models that is common across languages, and motivate further research into multilingual adversarial attacks.



## **50. Security and Privacy Issues of Federated Learning**

cs.CR

6 pages, 2 figures

**SubmitDate**: 2023-07-22    [abs](http://arxiv.org/abs/2307.12181v1) [paper-pdf](http://arxiv.org/pdf/2307.12181v1)

**Authors**: Jahid Hasan

**Abstract**: Federated Learning (FL) has emerged as a promising approach to address data privacy and confidentiality concerns by allowing multiple participants to construct a shared model without centralizing sensitive data. However, this decentralized paradigm introduces new security challenges, necessitating a comprehensive identification and classification of potential risks to ensure FL's security guarantees. This paper presents a comprehensive taxonomy of security and privacy challenges in Federated Learning (FL) across various machine learning models, including large language models. We specifically categorize attacks performed by the aggregator and participants, focusing on poisoning attacks, backdoor attacks, membership inference attacks, generative adversarial network (GAN) based attacks, and differential privacy attacks. Additionally, we propose new directions for future research, seeking innovative solutions to fortify FL systems against emerging security risks and uphold sensitive data confidentiality in distributed learning environments.



