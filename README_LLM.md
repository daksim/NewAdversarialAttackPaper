# Latest Large Language Model Attack Papers
**update at 2024-06-06 09:42:39**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03230v1) [paper-pdf](http://arxiv.org/pdf/2406.03230v1)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply an established methodology for analyzing distinctive activation patterns in the residual streams for a novel result of attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **2. Text Embedding Inversion Security for Multilingual Language Models**

cs.CL

18 pages, 17 Tables, 6 Figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2401.12192v4) [paper-pdf](http://arxiv.org/pdf/2401.12192v4)

**Authors**: Yiyi Chen, Heather Lent, Johannes Bjerva

**Abstract**: Textual data is often represented as real-numbered embeddings in NLP, particularly with the popularity of large language models (LLMs) and Embeddings as a Service (EaaS). However, storing sensitive information as embeddings can be susceptible to security breaches, as research shows that text can be reconstructed from embeddings, even without knowledge of the underlying model. While defence mechanisms have been explored, these are exclusively focused on English, leaving other languages potentially exposed to attacks. This work explores LLM security through multilingual embedding inversion. We define the problem of black-box multilingual and cross-lingual inversion attacks, and explore their potential implications. Our findings suggest that multilingual LLMs may be more vulnerable to inversion attacks, in part because English-based defences may be ineffective. To alleviate this, we propose a simple masking defense effective for both monolingual and multilingual models. This study is the first to investigate multilingual inversion attacks, shedding light on the differences in attacks and defenses across monolingual and multilingual settings.



## **3. BadAgent: Inserting and Activating Backdoor Attacks in LLM Agents**

cs.CL

Accepted by ACL 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03007v1) [paper-pdf](http://arxiv.org/pdf/2406.03007v1)

**Authors**: Yifei Wang, Dizhan Xue, Shengjie Zhang, Shengsheng Qian

**Abstract**: With the prosperity of large language models (LLMs), powerful LLM-based intelligent agents have been developed to provide customized services with a set of user-defined tools. State-of-the-art methods for constructing LLM agents adopt trained LLMs and further fine-tune them on data for the agent task. However, we show that such methods are vulnerable to our proposed backdoor attacks named BadAgent on various agent tasks, where a backdoor can be embedded by fine-tuning on the backdoor data. At test time, the attacker can manipulate the deployed LLM agents to execute harmful operations by showing the trigger in the agent input or environment. To our surprise, our proposed attack methods are extremely robust even after fine-tuning on trustworthy data. Though backdoor attacks have been studied extensively in natural language processing, to the best of our knowledge, we could be the first to study them on LLM agents that are more dangerous due to the permission to use external tools. Our work demonstrates the clear risk of constructing LLM agents based on untrusted LLMs or data. Our code is public at https://github.com/DPamK/BadAgent



## **4. Large Language Models are Few-shot Generators: Proposing Hybrid Prompt Algorithm To Generate Webshell Escape Samples**

cs.CR

21 pages, 17 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.07408v2) [paper-pdf](http://arxiv.org/pdf/2402.07408v2)

**Authors**: Mingrui Ma, Lansheng Han, Chunjie Zhou

**Abstract**: The frequent occurrence of cyber-attacks has made webshell attacks and defense gradually become a research hotspot in the field of network security. However, the lack of publicly available benchmark datasets and the over-reliance on manually defined rules for webshell escape sample generation have slowed down the progress of research related to webshell escape sample generation and artificial intelligence (AI)-based webshell detection. To address the drawbacks of weak webshell sample escape capabilities, the lack of webshell datasets with complex malicious features, and to promote the development of webshell detection, we propose the Hybrid Prompt algorithm for webshell escape sample generation with the help of large language models. As a prompt algorithm specifically developed for webshell sample generation, the Hybrid Prompt algorithm not only combines various prompt ideas including Chain of Thought, Tree of Thought, but also incorporates various components such as webshell hierarchical module and few-shot example to facilitate the LLM in learning and reasoning webshell escape strategies. Experimental results show that the Hybrid Prompt algorithm can work with multiple LLMs with excellent code reasoning ability to generate high-quality webshell samples with high Escape Rate (88.61% with GPT-4 model on VirusTotal detection engine) and (Survival Rate 54.98% with GPT-4 model).



## **5. Enhancing Jailbreak Attack Against Large Language Models through Silent Tokens**

cs.AI

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.20653v2) [paper-pdf](http://arxiv.org/pdf/2405.20653v2)

**Authors**: Jiahao Yu, Haozheng Luo, Jerry Yao-Chieh Hu, Wenbo Guo, Han Liu, Xinyu Xing

**Abstract**: Along with the remarkable successes of Language language models, recent research also started to explore the security threats of LLMs, including jailbreaking attacks. Attackers carefully craft jailbreaking prompts such that a target LLM will respond to the harmful question. Existing jailbreaking attacks require either human experts or leveraging complicated algorithms to craft jailbreaking prompts. In this paper, we introduce BOOST, a simple attack that leverages only the eos tokens. We demonstrate that rather than constructing complicated jailbreaking prompts, the attacker can simply append a few eos tokens to the end of a harmful question. It will bypass the safety alignment of LLMs and lead to successful jailbreaking attacks. We further apply BOOST to four representative jailbreak methods and show that the attack success rates of these methods can be significantly enhanced by simply adding eos tokens to the prompt. To understand this simple but novel phenomenon, we conduct empirical analyses. Our analysis reveals that adding eos tokens makes the target LLM believe the input is much less harmful, and eos tokens have low attention values and do not affect LLM's understanding of the harmful questions, leading the model to actually respond to the questions. Our findings uncover how fragile an LLM is against jailbreak attacks, motivating the development of strong safety alignment approaches.



## **6. Large Language Models Spot Phishing Emails with Surprising Accuracy: A Comparative Analysis of Performance**

cs.CL

7 pages, 3 figures

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2404.15485v2) [paper-pdf](http://arxiv.org/pdf/2404.15485v2)

**Authors**: Het Patel, Umair Rehman, Farkhund Iqbal

**Abstract**: Phishing, a prevalent cybercrime tactic for decades, remains a significant threat in today's digital world. By leveraging clever social engineering elements and modern technology, cybercrime targets many individuals, businesses, and organizations to exploit trust and security. These cyber-attackers are often disguised in many trustworthy forms to appear as legitimate sources. By cleverly using psychological elements like urgency, fear, social proof, and other manipulative strategies, phishers can lure individuals into revealing sensitive and personalized information. Building on this pervasive issue within modern technology, this paper aims to analyze the effectiveness of 15 Large Language Models (LLMs) in detecting phishing attempts, specifically focusing on a randomized set of "419 Scam" emails. The objective is to determine which LLMs can accurately detect phishing emails by analyzing a text file containing email metadata based on predefined criteria. The experiment concluded that the following models, ChatGPT 3.5, GPT-3.5-Turbo-Instruct, and ChatGPT, were the most effective in detecting phishing emails.



## **7. Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models**

cs.CL

ACL 2024 (main conference)

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.14007v2) [paper-pdf](http://arxiv.org/pdf/2402.14007v2)

**Authors**: Zhiwei He, Binglin Zhou, Hongkun Hao, Aiwei Liu, Xing Wang, Zhaopeng Tu, Zhuosheng Zhang, Rui Wang

**Abstract**: Text watermarking technology aims to tag and identify content produced by large language models (LLMs) to prevent misuse. In this study, we introduce the concept of cross-lingual consistency in text watermarking, which assesses the ability of text watermarks to maintain their effectiveness after being translated into other languages. Preliminary empirical results from two LLMs and three watermarking methods reveal that current text watermarking technologies lack consistency when texts are translated into various languages. Based on this observation, we propose a Cross-lingual Watermark Removal Attack (CWRA) to bypass watermarking by first obtaining a response from an LLM in a pivot language, which is then translated into the target language. CWRA can effectively remove watermarks, decreasing the AUCs to a random-guessing level without performance loss. Furthermore, we analyze two key factors that contribute to the cross-lingual consistency in text watermarking and propose X-SIR as a defense method against CWRA. Code: https://github.com/zwhe99/X-SIR.



## **8. QROA: A Black-Box Query-Response Optimization Attack on LLMs**

cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02044v1) [paper-pdf](http://arxiv.org/pdf/2406.02044v1)

**Authors**: Hussein Jawad, Nicolas J. -B. BRUNEL

**Abstract**: Large Language Models (LLMs) have surged in popularity in recent months, yet they possess concerning capabilities for generating harmful content when manipulated. This study introduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model's logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80\%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs.



## **9. Bileve: Securing Text Provenance in Large Language Models Against Spoofing with Bi-level Signature**

cs.CR

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01946v1) [paper-pdf](http://arxiv.org/pdf/2406.01946v1)

**Authors**: Tong Zhou, Xuandong Zhao, Xiaolin Xu, Shaolei Ren

**Abstract**: Text watermarks for large language models (LLMs) have been commonly used to identify the origins of machine-generated content, which is promising for assessing liability when combating deepfake or harmful content. While existing watermarking techniques typically prioritize robustness against removal attacks, unfortunately, they are vulnerable to spoofing attacks: malicious actors can subtly alter the meanings of LLM-generated responses or even forge harmful content, potentially misattributing blame to the LLM developer. To overcome this, we introduce a bi-level signature scheme, Bileve, which embeds fine-grained signature bits for integrity checks (mitigating spoofing attacks) as well as a coarse-grained signal to trace text sources when the signature is invalid (enhancing detectability) via a novel rank-based sampling strategy. Compared to conventional watermark detectors that only output binary results, Bileve can differentiate 5 scenarios during detection, reliably tracing text provenance and regulating LLMs. The experiments conducted on OPT-1.3B and LLaMA-7B demonstrate the effectiveness of Bileve in defeating spoofing attacks with enhanced detectability.



## **10. ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings**

cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.16006v2) [paper-pdf](http://arxiv.org/pdf/2402.16006v2)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.



## **11. HoneyGPT: Breaking the Trilemma in Terminal Honeypots with Large Language Model**

cs.CR

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01882v1) [paper-pdf](http://arxiv.org/pdf/2406.01882v1)

**Authors**: Ziyang Wang, Jianzhou You, Haining Wang, Tianwei Yuan, Shichao Lv, Yang Wang, Limin Sun

**Abstract**: Honeypots, as a strategic cyber-deception mechanism designed to emulate authentic interactions and bait unauthorized entities, continue to struggle with balancing flexibility, interaction depth, and deceptive capability despite their evolution over decades. Often they also lack the capability of proactively adapting to an attacker's evolving tactics, which restricts the depth of engagement and subsequent information gathering. Under this context, the emergent capabilities of large language models, in tandem with pioneering prompt-based engineering techniques, offer a transformative shift in the design and deployment of honeypot technologies. In this paper, we introduce HoneyGPT, a pioneering honeypot architecture based on ChatGPT, heralding a new era of intelligent honeypot solutions characterized by their cost-effectiveness, high adaptability, and enhanced interactivity, coupled with a predisposition for proactive attacker engagement. Furthermore, we present a structured prompt engineering framework that augments long-term interaction memory and robust security analytics. This framework, integrating thought of chain tactics attuned to honeypot contexts, enhances interactivity and deception, deepens security analytics, and ensures sustained engagement.   The evaluation of HoneyGPT includes two parts: a baseline comparison based on a collected dataset and a field evaluation in real scenarios for four weeks. The baseline comparison demonstrates HoneyGPT's remarkable ability to strike a balance among flexibility, interaction depth, and deceptive capability. The field evaluation further validates HoneyGPT's efficacy, showing its marked superiority in enticing attackers into more profound interactive engagements and capturing a wider array of novel attack vectors in comparison to existing honeypot technologies.



## **12. Safeguarding Large Language Models: A Survey**

cs.CR

under review. arXiv admin note: text overlap with arXiv:2402.01822

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.02622v1) [paper-pdf](http://arxiv.org/pdf/2406.02622v1)

**Authors**: Yi Dong, Ronghui Mu, Yanghao Zhang, Siqi Sun, Tianle Zhang, Changshun Wu, Gaojie Jin, Yi Qi, Jinwei Hu, Jie Meng, Saddek Bensalem, Xiaowei Huang

**Abstract**: In the burgeoning field of Large Language Models (LLMs), developing a robust safety mechanism, colloquially known as "safeguards" or "guardrails", has become imperative to ensure the ethical use of LLMs within prescribed boundaries. This article provides a systematic literature review on the current status of this critical mechanism. It discusses its major challenges and how it can be enhanced into a comprehensive mechanism dealing with ethical issues in various contexts. First, the paper elucidates the current landscape of safeguarding mechanisms that major LLM service providers and the open-source community employ. This is followed by the techniques to evaluate, analyze, and enhance some (un)desirable properties that a guardrail might want to enforce, such as hallucinations, fairness, privacy, and so on. Based on them, we review techniques to circumvent these controls (i.e., attacks), to defend the attacks, and to reinforce the guardrails. While the techniques mentioned above represent the current status and the active research trends, we also discuss several challenges that cannot be easily dealt with by the methods and present our vision on how to implement a comprehensive guardrail through the full consideration of multi-disciplinary approach, neural-symbolic method, and systems development lifecycle.



## **13. Here's a Free Lunch: Sanitizing Backdoored Models with Model Merge**

cs.CL

accepted to ACL2024 (Findings)

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2402.19334v2) [paper-pdf](http://arxiv.org/pdf/2402.19334v2)

**Authors**: Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, Qiongkai Xu

**Abstract**: The democratization of pre-trained language models through open-source initiatives has rapidly advanced innovation and expanded access to cutting-edge technologies. However, this openness also brings significant security risks, including backdoor attacks, where hidden malicious behaviors are triggered by specific inputs, compromising natural language processing (NLP) system integrity and reliability. This paper suggests that merging a backdoored model with other homogeneous models can significantly remediate backdoor vulnerabilities even if such models are not entirely secure. In our experiments, we verify our hypothesis on various models (BERT-Base, RoBERTa-Large, Llama2-7B, and Mistral-7B) and datasets (SST-2, OLID, AG News, and QNLI). Compared to multiple advanced defensive approaches, our method offers an effective and efficient inference-stage defense against backdoor attacks on classification and instruction-tuned tasks without additional resources or specific knowledge. Our approach consistently outperforms recent advanced baselines, leading to an average of about 75% reduction in the attack success rate. Since model merging has been an established approach for improving model performance, the extra advantage it provides regarding defense can be seen as a cost-free bonus.



## **14. Human vs. Machine: Behavioral Differences Between Expert Humans and Language Models in Wargame Simulations**

cs.CY

Updated with new plot and more details

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2403.03407v2) [paper-pdf](http://arxiv.org/pdf/2403.03407v2)

**Authors**: Max Lamparth, Anthony Corso, Jacob Ganz, Oriana Skylar Mastro, Jacquelyn Schneider, Harold Trinkunas

**Abstract**: To some, the advent of artificial intelligence (AI) promises better decision-making and increased military effectiveness while reducing the influence of human error and emotions. However, there is still debate about how AI systems, especially large language models (LLMs), behave compared to humans in high-stakes military decision-making scenarios with the potential for increased risks towards escalation and unnecessary conflicts. To test this potential and scrutinize the use of LLMs for such purposes, we use a new wargame experiment with 107 national security experts designed to look at crisis escalation in a fictional US-China scenario and compare human players to LLM-simulated responses in separate simulations. Wargames have a long history in the development of military strategy and the response of nations to threats or attacks. Here, we show a considerable high-level agreement in the LLM and human responses and significant quantitative and qualitative differences in individual actions and strategic tendencies. These differences depend on intrinsic biases in LLMs regarding the appropriate level of violence following strategic instructions, the choice of LLM, and whether the LLMs are tasked to decide for a team of players directly or first to simulate dialog between players. When simulating the dialog, the discussions lack quality and maintain a farcical harmony. The LLM simulations cannot account for human player characteristics, showing no significant difference even for extreme traits, such as "pacifist" or "aggressive sociopath". Our results motivate policymakers to be cautious before granting autonomy or following AI-based strategy recommendations.



## **15. PrivacyRestore: Privacy-Preserving Inference in Large Language Models via Privacy Removal and Restoration**

cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01394v1) [paper-pdf](http://arxiv.org/pdf/2406.01394v1)

**Authors**: Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Huiping Zhuang, Cen Chen

**Abstract**: The widespread usage of online Large Language Models (LLMs) inference services has raised significant privacy concerns about the potential exposure of private information in user inputs to eavesdroppers or untrustworthy service providers. Existing privacy protection methods for LLMs suffer from insufficient privacy protection, performance degradation, or severe inference time overhead. In this paper, we propose PrivacyRestore to protect the privacy of user inputs during LLM inference. PrivacyRestore directly removes privacy spans in user inputs and restores privacy information via activation steering during inference. The privacy spans are encoded as restoration vectors. We propose Attention-aware Weighted Aggregation (AWA) which aggregates restoration vectors of all privacy spans in the input into a meta restoration vector. AWA not only ensures proper representation of all privacy spans but also prevents attackers from inferring the privacy spans from the meta restoration vector alone. This meta restoration vector, along with the query with privacy spans removed, is then sent to the server. The experimental results show that PrivacyRestore can protect private information while maintaining acceptable levels of performance and inference efficiency.



## **16. Privacy in LLM-based Recommendation: Recent Advances and Future Directions**

cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01363v1) [paper-pdf](http://arxiv.org/pdf/2406.01363v1)

**Authors**: Sichun Luo, Wei Shao, Yuxuan Yao, Jian Xu, Mingyang Liu, Qintong Li, Bowei He, Maolin Wang, Guanzhi Deng, Hanxu Hou, Xinyi Zhang, Linqi Song

**Abstract**: Nowadays, large language models (LLMs) have been integrated with conventional recommendation models to improve recommendation performance. However, while most of the existing works have focused on improving the model performance, the privacy issue has only received comparatively less attention. In this paper, we review recent advancements in privacy within LLM-based recommendation, categorizing them into privacy attacks and protection mechanisms. Additionally, we highlight several challenges and propose future directions for the community to address these critical problems.



## **17. DepsRAG: Towards Managing Software Dependencies using Large Language Models**

cs.SE

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.20455v2) [paper-pdf](http://arxiv.org/pdf/2405.20455v2)

**Authors**: Mohannad Alhanahnah, Yazan Boshmaf, Benoit Baudry

**Abstract**: Managing software dependencies is a crucial maintenance task in software development and is becoming a rapidly growing research field, especially in light of the significant increase in software supply chain attacks. Specialized expertise and substantial developer effort are required to fully comprehend dependencies and reveal hidden properties about the dependencies (e.g., number of dependencies, dependency chains, depth of dependencies).   Recent advancements in Large Language Models (LLMs) allow the retrieval of information from various data sources for response generation, thus providing a new opportunity to uniquely manage software dependencies. To highlight the potential of this technology, we present~\tool, a proof-of-concept Retrieval Augmented Generation (RAG) approach that constructs direct and transitive dependencies of software packages as a Knowledge Graph (KG) in four popular software ecosystems. DepsRAG can answer user questions about software dependencies by automatically generating necessary queries to retrieve information from the KG, and then augmenting the input of LLMs with the retrieved information. DepsRAG can also perform Web search to answer questions that the LLM cannot directly answer via the KG. We identify tangible benefits that DepsRAG can offer and discuss its limitations.



## **18. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

cs.MM

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.19802v2) [paper-pdf](http://arxiv.org/pdf/2405.19802v2)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.



## **19. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11082v6) [paper-pdf](http://arxiv.org/pdf/2304.11082v6)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **20. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01179v1) [paper-pdf](http://arxiv.org/pdf/2406.01179v1)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.



## **21. Genshin: General Shield for Natural Language Processing with Large Language Models**

cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.18741v2) [paper-pdf](http://arxiv.org/pdf/2405.18741v2)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.



## **22. Data Contamination Calibration for Black-box LLMs**

cs.LG

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.11930v2) [paper-pdf](http://arxiv.org/pdf/2405.11930v2)

**Authors**: Wentao Ye, Jiaqi Hu, Liyao Li, Haobo Wang, Gang Chen, Junbo Zhao

**Abstract**: The rapid advancements of Large Language Models (LLMs) tightly associate with the expansion of the training data size. However, the unchecked ultra-large-scale training sets introduce a series of potential risks like data contamination, i.e. the benchmark data is used for training. In this work, we propose a holistic method named Polarized Augment Calibration (PAC) along with a new to-be-released dataset to detect the contaminated data and diminish the contamination effect. PAC extends the popular MIA (Membership Inference Attack) -- from machine learning community -- by forming a more global target at detecting training data to Clarify invisible training data. As a pioneering work, PAC is very much plug-and-play that can be integrated with most (if not all) current white- and black-box LLMs. By extensive experiments, PAC outperforms existing methods by at least 4.5%, towards data contamination detection on more 4 dataset formats, with more than 10 base LLMs. Besides, our application in real-world scenarios highlights the prominent presence of contamination and related issues.



## **23. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.00083v1) [paper-pdf](http://arxiv.org/pdf/2406.00083v1)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.



## **24. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

cs.CL

Computational Linguistics. Submitted manuscript.  https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00519/121095/Cross-lingual-Cross-temporal-Summarization-Dataset

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2306.12916v3) [paper-pdf](http://arxiv.org/pdf/2306.12916v3)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We (1) build the first CLCTS corpus with 328 instances for hDe-En (extended version with 455 instances) and 289 for hEn-De (extended version with 501 instances), leveraging historical fiction texts and Wikipedia summaries in English and German; (2) examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks; (3) explore the potential of GPT-3.5 as a summarizer; (4) report evaluations from humans, GPT-4, and several recent automatic evaluation metrics. Our results indicate that intermediate task finetuned end-to-end models generate bad to moderate quality summaries while GPT-3.5, as a zero-shot summarizer, provides moderate to good quality outputs. GPT-3.5 also seems very adept at normalizing historical text. To assess data contamination in GPT-3.5, we design an adversarial attack scheme in which we find that GPT-3.5 performs slightly worse for unseen source documents compared to seen documents. Moreover, it sometimes hallucinates when the source sentences are inverted against its prior knowledge with a summarization accuracy of 0.67 for plot omission, 0.71 for entity swap, and 0.53 for plot negation. Overall, our regression results of model performances suggest that longer, older, and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task.



## **25. Are you still on track!? Catching LLM Task Drift with Activations**

cs.CR

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2406.00799v1) [paper-pdf](http://arxiv.org/pdf/2406.00799v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.



## **26. Transforming Computer Security and Public Trust Through the Exploration of Fine-Tuning Large Language Models**

cs.CL

A preprint, 17 pages. 11 images

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2406.00628v1) [paper-pdf](http://arxiv.org/pdf/2406.00628v1)

**Authors**: Garrett Crumrine, Izzat Alsmadi, Jesus Guerrero, Yuvaraj Munian

**Abstract**: Large language models (LLMs) have revolutionized how we interact with machines. However, this technological advancement has been paralleled by the emergence of "Mallas," malicious services operating underground that exploit LLMs for nefarious purposes. Such services create malware, phishing attacks, and deceptive websites, escalating the cyber security threats landscape. This paper delves into the proliferation of Mallas by examining the use of various pre-trained language models and their efficiency and vulnerabilities when misused. Building on a dataset from the Common Vulnerabilities and Exposures (CVE) program, it explores fine-tuning methodologies to generate code and explanatory text related to identified vulnerabilities. This research aims to shed light on the operational strategies and exploitation techniques of Mallas, leading to the development of more secure and trustworthy AI applications. The paper concludes by emphasizing the need for further research, enhanced safeguards, and ethical guidelines to mitigate the risks associated with the malicious application of LLMs.



## **27. Exploring Vulnerabilities and Protections in Large Language Models: A Survey**

cs.LG

**SubmitDate**: 2024-06-01    [abs](http://arxiv.org/abs/2406.00240v1) [paper-pdf](http://arxiv.org/pdf/2406.00240v1)

**Authors**: Frank Weizhen Liu, Chenhui Hu

**Abstract**: As Large Language Models (LLMs) increasingly become key components in various AI applications, understanding their security vulnerabilities and the effectiveness of defense mechanisms is crucial. This survey examines the security challenges of LLMs, focusing on two main areas: Prompt Hacking and Adversarial Attacks, each with specific types of threats. Under Prompt Hacking, we explore Prompt Injection and Jailbreaking Attacks, discussing how they work, their potential impacts, and ways to mitigate them. Similarly, we analyze Adversarial Attacks, breaking them down into Data Poisoning Attacks and Backdoor Attacks. This structured examination helps us understand the relationships between these vulnerabilities and the defense strategies that can be implemented. The survey highlights these security challenges and discusses robust defensive frameworks to protect LLMs against these threats. By detailing these security issues, the survey contributes to the broader discussion on creating resilient AI systems that can resist sophisticated attacks.



## **28. ReEval: Automatic Hallucination Evaluation for Retrieval-Augmented Large Language Models via Transferable Adversarial Attacks**

cs.CL

NAACL 2024 Findings

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2310.12516v2) [paper-pdf](http://arxiv.org/pdf/2310.12516v2)

**Authors**: Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao

**Abstract**: Despite remarkable advancements in mitigating hallucinations in large language models (LLMs) by retrieval augmentation, it remains challenging to measure the reliability of LLMs using static question-answering (QA) data. Specifically, given the potential of data contamination (e.g., leading to memorization), good static benchmark performance does not ensure that model can reliably use the provided evidence for responding, which is essential to avoid hallucination when the required knowledge is new or private. Inspired by adversarial machine learning, we investigate the feasibility of automatically perturbing existing static one for dynamic evaluation. Specifically, this paper presents ReEval, an LLM-based framework using prompt chaining to perturb the original evidence for generating new test cases for evaluating the LLMs' reliability in using new evidence for answering.   We implement ReEval using ChatGPT and evaluate the resulting variants of two popular open-domain QA datasets on a collection of LLMs under various prompting settings. Our generated data is human-readable and useful to trigger hallucination in LLM. Accurate models on static data are observed to produce unsupported answers from the perturbed evidence, with pronounced accuracy drops across LLMs including GPT-4. We find that our adversarial examples are transferable across all considered LLMs. The examples generated by a small model can be used to evaluate a much larger model, making our approach cost-effective.



## **29. Improved Techniques for Optimization-Based Jailbreaking on Large Language Models**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.21018v2) [paper-pdf](http://arxiv.org/pdf/2405.21018v2)

**Authors**: Xiaojun Jia, Tianyu Pang, Chao Du, Yihao Huang, Jindong Gu, Yang Liu, Xiaochun Cao, Min Lin

**Abstract**: Large language models (LLMs) are being rapidly developed, and a key component of their widespread deployment is their safety-related alignment. Many red-teaming efforts aim to jailbreak LLMs, where among these efforts, the Greedy Coordinate Gradient (GCG) attack's success has led to a growing interest in the study of optimization-based jailbreaking techniques. Although GCG is a significant milestone, its attacking efficiency remains unsatisfactory. In this paper, we present several improved (empirical) techniques for optimization-based jailbreaks like GCG. We first observe that the single target template of "Sure" largely limits the attacking performance of GCG; given this, we propose to apply diverse target templates containing harmful self-suggestion and/or guidance to mislead LLMs. Besides, from the optimization aspects, we propose an automatic multi-coordinate updating strategy in GCG (i.e., adaptively deciding how many tokens to replace in each step) to accelerate convergence, as well as tricks like easy-to-hard initialisation. Then, we combine these improved technologies to develop an efficient jailbreak method, dubbed I-GCG. In our experiments, we evaluate on a series of benchmarks (such as NeurIPS 2023 Red Teaming Track). The results demonstrate that our improved techniques can help GCG outperform state-of-the-art jailbreaking attacks and achieve nearly 100% attack success rate. The code is released at https://github.com/jiaxiaojunQAQ/I-GCG.



## **30. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.13401v3) [paper-pdf](http://arxiv.org/pdf/2405.13401v3)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.



## **31. Preemptive Answer "Attacks" on Chain-of-Thought Reasoning**

cs.CL

Accepted to ACL'24 (Findings). Camera-ready version

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.20902v1) [paper-pdf](http://arxiv.org/pdf/2405.20902v1)

**Authors**: Rongwu Xu, Zehan Qi, Wei Xu

**Abstract**: Large language models (LLMs) showcase impressive reasoning capabilities when coupled with Chain-of-Thought (CoT) prompting. However, the robustness of this approach warrants further investigation. In this paper, we introduce a novel scenario termed preemptive answers, where the LLM obtains an answer before engaging in reasoning. This situation can arise inadvertently or induced by malicious users by prompt injection attacks. Experiments reveal that preemptive answers significantly impair the model's reasoning capability across various CoT methods and a broad spectrum of datasets. To bolster the robustness of reasoning, we propose two measures aimed at mitigating this issue to some extent.



## **32. Robustifying Safety-Aligned Large Language Models through Clean Data Curation**

cs.CR

**SubmitDate**: 2024-05-31    [abs](http://arxiv.org/abs/2405.19358v2) [paper-pdf](http://arxiv.org/pdf/2405.19358v2)

**Authors**: Xiaoqun Liu, Jiacheng Liang, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are vulnerable when trained on datasets containing harmful content, which leads to potential jailbreaking attacks in two scenarios: the integration of harmful texts within crowdsourced data used for pre-training and direct tampering with LLMs through fine-tuning. In both scenarios, adversaries can compromise the safety alignment of LLMs, exacerbating malfunctions. Motivated by the need to mitigate these adversarial influences, our research aims to enhance safety alignment by either neutralizing the impact of malicious texts in pre-training datasets or increasing the difficulty of jailbreaking during downstream fine-tuning. In this paper, we propose a data curation framework designed to counter adversarial impacts in both scenarios. Our method operates under the assumption that we have no prior knowledge of attack details, focusing solely on curating clean texts. We introduce an iterative process aimed at revising texts to reduce their perplexity as perceived by LLMs, while simultaneously preserving their text quality. By pre-training or fine-tuning LLMs with curated clean texts, we observe a notable improvement in LLM robustness regarding safety alignment against harmful queries. For instance, when pre-training LLMs using a crowdsourced dataset containing 5\% harmful instances, adding an equivalent amount of curated texts significantly mitigates the likelihood of providing harmful responses in LLMs and reduces the attack success rate by 71\%. Our study represents a significant step towards mitigating the risks associated with training-based jailbreaking and fortifying the secure utilization of LLMs.



## **33. Phantom: General Trigger Attacks on Retrieval Augmented Language Generation**

cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20485v1) [paper-pdf](http://arxiv.org/pdf/2405.20485v1)

**Authors**: Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs) in chatbot applications, enabling developers to adapt and personalize the LLM output without expensive training or fine-tuning. RAG systems use an external knowledge database to retrieve the most relevant documents for a given query, providing this context to the LLM generator. While RAG achieves impressive utility in many applications, its adoption to enable personalized generative models introduces new security risks. In this work, we propose new attack surfaces for an adversary to compromise a victim's RAG system, by injecting a single malicious document in its knowledge database. We design Phantom, general two-step attack framework against RAG augmented LLMs. The first step involves crafting a poisoned document designed to be retrieved by the RAG system within the top-k results only when an adversarial trigger, a specific sequence of words acting as backdoor, is present in the victim's queries. In the second step, a specially crafted adversarial string within the poisoned document triggers various adversarial attacks in the LLM generator, including denial of service, reputation damage, privacy violations, and harmful behaviors. We demonstrate our attacks on multiple LLM architectures, including Gemma, Vicuna, and Llama.



## **34. Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters**

cs.CR

20 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20413v1) [paper-pdf](http://arxiv.org/pdf/2405.20413v1)

**Authors**: Haibo Jin, Andy Zhou, Joe D. Menke, Haohan Wang

**Abstract**: Large Language Models (LLMs) are typically harmless but remain vulnerable to carefully crafted prompts known as ``jailbreaks'', which can bypass protective measures and induce harmful behavior. Recent advancements in LLMs have incorporated moderation guardrails that can filter outputs, which trigger processing errors for certain malicious questions. Existing red-teaming benchmarks often neglect to include questions that trigger moderation guardrails, making it difficult to evaluate jailbreak effectiveness. To address this issue, we introduce JAMBench, a harmful behavior benchmark designed to trigger and evaluate moderation guardrails. JAMBench involves 160 manually crafted instructions covering four major risk categories at multiple severity levels. Furthermore, we propose a jailbreak method, JAM (Jailbreak Against Moderation), designed to attack moderation guardrails using jailbreak prefixes to bypass input-level filters and a fine-tuned shadow model functionally equivalent to the guardrail model to generate cipher characters to bypass output-level filters. Our extensive experiments on four LLMs demonstrate that JAM achieves higher jailbreak success ($\sim$ $\times$ 19.88) and lower filtered-out rates ($\sim$ $\times$ 1/6) than baselines.



## **35. Context Injection Attacks on Large Language Models**

cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20234v1) [paper-pdf](http://arxiv.org/pdf/2405.20234v1)

**Authors**: Cheng'an Wei, Kai Chen, Yue Zhao, Yujia Gong, Lu Xiang, Shenchen Zhu

**Abstract**: Large Language Models (LLMs) such as ChatGPT and Llama-2 have become prevalent in real-world applications, exhibiting impressive text generation performance. LLMs are fundamentally developed from a scenario where the input data remains static and lacks a clear structure. To behave interactively over time, LLM-based chat systems must integrate additional contextual information (i.e., chat history) into their inputs, following a pre-defined structure. This paper identifies how such integration can expose LLMs to misleading context from untrusted sources and fail to differentiate between system and user inputs, allowing users to inject context. We present a systematic methodology for conducting context injection attacks aimed at eliciting disallowed responses by introducing fabricated context. This could lead to illegal actions, inappropriate content, or technology misuse. Our context fabrication strategies, acceptance elicitation and word anonymization, effectively create misleading contexts that can be structured with attacker-customized prompt templates, achieving injection through malicious user messages. Comprehensive evaluations on real-world LLMs such as ChatGPT and Llama-2 confirm the efficacy of the proposed attack with success rates reaching 97%. We also discuss potential countermeasures that can be adopted for attack detection and developing more secure models. Our findings provide insights into the challenges associated with the real-world deployment of LLMs for interactive and structured data scenarios.



## **36. Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20099v1) [paper-pdf](http://arxiv.org/pdf/2405.20099v1)

**Authors**: Chen Xiong, Xiangyu Qi, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Safety, security, and compliance are essential requirements when aligning large language models (LLMs). However, many seemingly aligned LLMs are soon shown to be susceptible to jailbreak attacks. These attacks aim to circumvent the models' safety guardrails and security mechanisms by introducing jailbreak prompts into malicious queries. In response to these challenges, this paper introduces Defensive Prompt Patch (DPP), a novel prompt-based defense mechanism specifically designed to protect LLMs against such sophisticated jailbreak strategies. Unlike previous approaches, which have often compromised the utility of the model for the sake of safety, DPP is designed to achieve a minimal Attack Success Rate (ASR) while preserving the high utility of LLMs. Our method uses strategically designed interpretable suffix prompts that effectively thwart a wide range of standard and adaptive jailbreak techniques. Empirical results conducted on LLAMA-2-7B-Chat and Mistral-7B-Instruct-v0.2 models demonstrate the robustness and adaptability of DPP, showing significant reductions in ASR with negligible impact on utility. Our approach not only outperforms existing defense strategies in balancing safety and functionality, but also provides a scalable and interpretable solution applicable to various LLM platforms.



## **37. Typography Leads Semantic Diversifying: Amplifying Adversarial Transferability across Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20090v1) [paper-pdf](http://arxiv.org/pdf/2405.20090v1)

**Authors**: Hao Cheng, Erjia Xiao, Jiahang Cao, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Following the advent of the Artificial Intelligence (AI) era of large models, Multimodal Large Language Models (MLLMs) with the ability to understand cross-modal interactions between vision and text have attracted wide attention. Adversarial examples with human-imperceptible perturbation are shown to possess a characteristic known as transferability, which means that a perturbation generated by one model could also mislead another different model. Augmenting the diversity in input data is one of the most significant methods for enhancing adversarial transferability. This method has been certified as a way to significantly enlarge the threat impact under black-box conditions. Research works also demonstrate that MLLMs can be exploited to generate adversarial examples in the white-box scenario. However, the adversarial transferability of such perturbations is quite limited, failing to achieve effective black-box attacks across different models. In this paper, we propose the Typographic-based Semantic Transfer Attack (TSTA), which is inspired by: (1) MLLMs tend to process semantic-level information; (2) Typographic Attack could effectively distract the visual information captured by MLLMs. In the scenarios of Harmful Word Insertion and Important Information Protection, our TSTA demonstrates superior performance.



## **38. Efficient LLM-Jailbreaking by Introducing Visual Modality**

cs.AI

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.20015v1) [paper-pdf](http://arxiv.org/pdf/2405.20015v1)

**Authors**: Zhenxing Niu, Yuyao Sun, Haodong Ren, Haoxuan Ji, Quan Wang, Xiaoke Ma, Gang Hua, Rong Jin

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreaks that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) through the incorporation of a visual module into the target LLM. Subsequently, we conduct an efficient MLLM-jailbreak to generate jailbreaking embeddings embJS. Finally, we convert the embJS into text space to facilitate the jailbreaking of the target LLM. Compared to direct LLM-jailbreaking, our approach is more efficient, as MLLMs are more vulnerable to jailbreaking than pure LLM. Additionally, to improve the attack success rate (ASR) of jailbreaking, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class jailbreaking capabilities.



## **39. Vocabulary Attack to Hijack Large Language Model Applications**

cs.CR

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2404.02637v2) [paper-pdf](http://arxiv.org/pdf/2404.02637v2)

**Authors**: Patrick Levi, Christoph P. Neumann

**Abstract**: The fast advancements in Large Language Models (LLMs) are driving an increasing number of applications. Together with the growing number of users, we also see an increasing number of attackers who try to outsmart these systems. They want the model to reveal confidential information, specific false information, or offensive behavior. To this end, they manipulate their instructions for the LLM by inserting separators or rephrasing them systematically until they reach their goal. Our approach is different. It inserts words from the model vocabulary. We find these words using an optimization procedure and embeddings from another LLM (attacker LLM). We prove our approach by goal hijacking two popular open-source LLMs from the Llama2 and the Flan-T5 families, respectively. We present two main findings. First, our approach creates inconspicuous instructions and therefore it is hard to detect. For many attack cases, we find that even a single word insertion is sufficient. Second, we demonstrate that we can conduct our attack using a different model than the target model to conduct our attack with.



## **40. Large Language Model Watermark Stealing With Mixed Integer Programming**

cs.CR

12 pages

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19677v1) [paper-pdf](http://arxiv.org/pdf/2405.19677v1)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, Leo Yu Zhang, Chao Chen, Shengshan Hu, Asif Gill, Shirui Pan

**Abstract**: The Large Language Model (LLM) watermark is a newly emerging technique that shows promise in addressing concerns surrounding LLM copyright, monitoring AI-generated text, and preventing its misuse. The LLM watermark scheme commonly includes generating secret keys to partition the vocabulary into green and red lists, applying a perturbation to the logits of tokens in the green list to increase their sampling likelihood, thus facilitating watermark detection to identify AI-generated text if the proportion of green tokens exceeds a threshold. However, recent research indicates that watermarking methods using numerous keys are susceptible to removal attacks, such as token editing, synonym substitution, and paraphrasing, with robustness declining as the number of keys increases. Therefore, the state-of-the-art watermark schemes that employ fewer or single keys have been demonstrated to be more robust against text editing and paraphrasing. In this paper, we propose a novel green list stealing attack against the state-of-the-art LLM watermark scheme and systematically examine its vulnerability to this attack. We formalize the attack as a mixed integer programming problem with constraints. We evaluate our attack under a comprehensive threat model, including an extreme scenario where the attacker has no prior knowledge, lacks access to the watermark detector API, and possesses no information about the LLM's parameter settings or watermark injection/detection scheme. Extensive experiments on LLMs, such as OPT and LLaMA, demonstrate that our attack can successfully steal the green list and remove the watermark across all settings.



## **41. AutoBreach: Universal and Adaptive Jailbreaking with Efficient Wordplay-Guided Optimization**

cs.CV

Under review

**SubmitDate**: 2024-05-30    [abs](http://arxiv.org/abs/2405.19668v1) [paper-pdf](http://arxiv.org/pdf/2405.19668v1)

**Authors**: Jiawei Chen, Xiao Yang, Zhengwei Fang, Yu Tian, Yinpeng Dong, Zhaoxia Yin, Hang Su

**Abstract**: Despite the widespread application of large language models (LLMs) across various tasks, recent studies indicate that they are susceptible to jailbreak attacks, which can render their defense mechanisms ineffective. However, previous jailbreak research has frequently been constrained by limited universality, suboptimal efficiency, and a reliance on manual crafting. In response, we rethink the approach to jailbreaking LLMs and formally define three essential properties from the attacker' s perspective, which contributes to guiding the design of jailbreak methods. We further introduce AutoBreach, a novel method for jailbreaking LLMs that requires only black-box access. Inspired by the versatility of wordplay, AutoBreach employs a wordplay-guided mapping rule sampling strategy to generate a variety of universal mapping rules for creating adversarial prompts. This generation process leverages LLMs' automatic summarization and reasoning capabilities, thus alleviating the manual burden. To boost jailbreak success rates, we further suggest sentence compression and chain-of-thought-based mapping rules to correct errors and wordplay misinterpretations in target LLMs. Additionally, we propose a two-stage mapping rule optimization strategy that initially optimizes mapping rules before querying target LLMs to enhance the efficiency of AutoBreach. AutoBreach can efficiently identify security vulnerabilities across various LLMs, including three proprietary models: Claude-3, GPT-3.5, GPT-4 Turbo, and two LLMs' web platforms: Bingchat, GPT-4 Web, achieving an average success rate of over 80% with fewer than 10 queries



## **42. Unlearning Climate Misinformation in Large Language Models**

cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19563v1) [paper-pdf](http://arxiv.org/pdf/2405.19563v1)

**Authors**: Michael Fore, Simranjit Singh, Chaehong Lee, Amritanshu Pandey, Antonios Anastasopoulos, Dimitrios Stamoulis

**Abstract**: Misinformation regarding climate change is a key roadblock in addressing one of the most serious threats to humanity. This paper investigates factual accuracy in large language models (LLMs) regarding climate information. Using true/false labeled Q&A data for fine-tuning and evaluating LLMs on climate-related claims, we compare open-source models, assessing their ability to generate truthful responses to climate change questions. We investigate the detectability of models intentionally poisoned with false climate information, finding that such poisoning may not affect the accuracy of a model's responses in other domains. Furthermore, we compare the effectiveness of unlearning algorithms, fine-tuning, and Retrieval-Augmented Generation (RAG) for factually grounding LLMs on climate change topics. Our evaluation reveals that unlearning algorithms can be effective for nuanced conceptual claims, despite previous findings suggesting their inefficacy in privacy contexts. These insights aim to guide the development of more factually reliable LLMs and highlight the need for additional work to secure LLMs against misinformation attacks.



## **43. Voice Jailbreak Attacks Against GPT-4o**

cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19103v1) [paper-pdf](http://arxiv.org/pdf/2405.19103v1)

**Authors**: Xinyue Shen, Yixin Wu, Michael Backes, Yang Zhang

**Abstract**: Recently, the concept of artificial assistants has evolved from science fiction into real-world applications. GPT-4o, the newest multimodal large language model (MLLM) across audio, vision, and text, has further blurred the line between fiction and reality by enabling more natural human-computer interactions. However, the advent of GPT-4o's voice mode may also introduce a new attack surface. In this paper, we present the first systematic measurement of jailbreak attacks against the voice mode of GPT-4o. We show that GPT-4o demonstrates good resistance to forbidden questions and text jailbreak prompts when directly transferring them to voice mode. This resistance is primarily due to GPT-4o's internal safeguards and the difficulty of adapting text jailbreak prompts to voice mode. Inspired by GPT-4o's human-like behaviors, we propose VoiceJailbreak, a novel voice jailbreak attack that humanizes GPT-4o and attempts to persuade it through fictional storytelling (setting, character, and plot). VoiceJailbreak is capable of generating simple, audible, yet effective jailbreak prompts, which significantly increases the average attack success rate (ASR) from 0.033 to 0.778 in six forbidden scenarios. We also conduct extensive experiments to explore the impacts of interaction steps, key elements of fictional writing, and different languages on VoiceJailbreak's effectiveness and further enhance the attack performance with advanced fictional writing techniques. We hope our study can assist the research community in building more secure and well-regulated MLLMs.



## **44. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.



## **45. DiveR-CT: Diversity-enhanced Red Teaming with Relaxing Constraints**

cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19026v1) [paper-pdf](http://arxiv.org/pdf/2405.19026v1)

**Authors**: Andrew Zhao, Quentin Xu, Matthieu Lin, Shenzhi Wang, Yong-jin Liu, Zilong Zheng, Gao Huang

**Abstract**: Recent advances in large language models (LLMs) have made them indispensable, raising significant concerns over managing their safety. Automated red teaming offers a promising alternative to the labor-intensive and error-prone manual probing for vulnerabilities, providing more consistent and scalable safety evaluations. However, existing approaches often compromise diversity by focusing on maximizing attack success rate. Additionally, methods that decrease the cosine similarity from historical embeddings with semantic diversity rewards lead to novelty stagnation as history grows. To address these issues, we introduce DiveR-CT, which relaxes conventional constraints on the objective and semantic reward, granting greater freedom for the policy to enhance diversity. Our experiments demonstrate DiveR-CT's marked superiority over baselines by 1) generating data that perform better in various diversity metrics across different attack success rate levels, 2) better-enhancing resiliency in blue team models through safety tuning based on collected data, 3) allowing dynamic control of objective weights for reliable and controllable attack success rates, and 4) reducing susceptibility to reward overoptimization. Project details and code can be found at https://andrewzh112.github.io/#diverct.



## **46. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.12523v2) [paper-pdf](http://arxiv.org/pdf/2405.12523v2)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.



## **47. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.



## **48. Learning diverse attacks on large language models for robust red-teaming and safety tuning**

cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18540v1) [paper-pdf](http://arxiv.org/pdf/2405.18540v1)

**Authors**: Seanie Lee, Minsu Kim, Lynn Cherif, David Dobre, Juho Lee, Sung Ju Hwang, Kenji Kawaguchi, Gauthier Gidel, Yoshua Bengio, Nikolay Malkin, Moksh Jain

**Abstract**: Red-teaming, or identifying prompts that elicit harmful responses, is a critical step in ensuring the safe and responsible deployment of large language models (LLMs). Developing effective protection against many modes of attack prompts requires discovering diverse attacks. Automated red-teaming typically uses reinforcement learning to fine-tune an attacker language model to generate prompts that elicit undesirable responses from a target LLM, as measured, for example, by an auxiliary toxicity classifier. We show that even with explicit regularization to favor novelty and diversity, existing approaches suffer from mode collapse or fail to generate effective attacks. As a flexible and probabilistically principled alternative, we propose to use GFlowNet fine-tuning, followed by a secondary smoothing phase, to train the attacker model to generate diverse and effective attack prompts. We find that the attacks generated by our method are effective against a wide range of target LLMs, both with and without safety tuning, and transfer well between target LLMs. Finally, we demonstrate that models safety-tuned using a dataset of red-teaming prompts generated by our method are robust to attacks from other RL-based red-teaming approaches.



## **49. Unleashing the potential of prompt engineering: a comprehensive review**

cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **50. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.



