# Latest Large Language Model Attack Papers
**update at 2025-02-12 10:56:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models**

cs.AI

16 pages. arXiv admin note: text overlap with arXiv:2404.04306

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07644v1) [paper-pdf](http://arxiv.org/pdf/2502.07644v1)

**Authors**: Shihao Xia, Mengting He, Shuai Shao, Tingting Yu, Yiying Zhang, Linhai Song

**Abstract**: To govern smart contracts running on Ethereum, multiple Ethereum Request for Comment (ERC) standards have been developed, each having a set of rules to guide the behaviors of smart contracts. Violating the ERC rules could cause serious security issues and financial loss, signifying the importance of verifying smart contracts follow ERCs. Today's practices of such verification are to manually audit each single contract, use expert-developed program-analysis tools, or use large language models (LLMs), all of which are far from effective in identifying ERC rule violations. This paper introduces SymGPT, a tool that combines the natural language understanding of large language models (LLMs) with the formal guarantees of symbolic execution to automatically verify smart contracts' compliance with ERC rules. To develop SymGPT, we conduct an empirical study of 132 ERC rules from three widely used ERC standards, examining their content, security implications, and natural language descriptions. Based on this study, we design SymGPT by first instructing an LLM to translate ERC rules into a defined EBNF grammar. We then synthesize constraints from the formalized rules to represent scenarios where violations may occur and use symbolic execution to detect them. Our evaluation shows that SymGPT identifies 5,783 ERC rule violations in 4,000 real-world contracts, including 1,375 violations with clear attack paths for stealing financial assets, demonstrating its effectiveness. Furthermore, SymGPT outperforms six automated techniques and a security-expert auditing service, underscoring its superiority over current smart contract analysis methods.



## **2. JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation**

cs.CR

To Appear in the 34rd USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07557v1) [paper-pdf](http://arxiv.org/pdf/2502.07557v1)

**Authors**: Shenyi Zhang, Yuchen Zhai, Keyan Guo, Hongxin Hu, Shengnan Guo, Zheng Fang, Lingchen Zhao, Chao Shen, Cong Wang, Qian Wang

**Abstract**: Despite the implementation of safety alignment strategies, large language models (LLMs) remain vulnerable to jailbreak attacks, which undermine these safety guardrails and pose significant security threats. Some defenses have been proposed to detect or mitigate jailbreaks, but they are unable to withstand the test of time due to an insufficient understanding of jailbreak mechanisms. In this work, we investigate the mechanisms behind jailbreaks based on the Linear Representation Hypothesis (LRH), which states that neural networks encode high-level concepts as subspaces in their hidden representations. We define the toxic semantics in harmful and jailbreak prompts as toxic concepts and describe the semantics in jailbreak prompts that manipulate LLMs to comply with unsafe requests as jailbreak concepts. Through concept extraction and analysis, we reveal that LLMs can recognize the toxic concepts in both harmful and jailbreak prompts. However, unlike harmful prompts, jailbreak prompts activate the jailbreak concepts and alter the LLM output from rejection to compliance. Building on our analysis, we propose a comprehensive jailbreak defense framework, JBShield, consisting of two key components: jailbreak detection JBShield-D and mitigation JBShield-M. JBShield-D identifies jailbreak prompts by determining whether the input activates both toxic and jailbreak concepts. When a jailbreak prompt is detected, JBShield-M adjusts the hidden representations of the target LLM by enhancing the toxic concept and weakening the jailbreak concept, ensuring LLMs produce safe content. Extensive experiments demonstrate the superior performance of JBShield, achieving an average detection accuracy of 0.95 and reducing the average attack success rate of various jailbreak attacks to 2% from 61% across distinct LLMs.



## **3. LUNAR: LLM Unlearning via Neural Activation Redirection**

cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.



## **4. LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild**

cs.CR

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.13919v2) [paper-pdf](http://arxiv.org/pdf/2410.13919v2)

**Authors**: Reworr, Dmitrii Volkov

**Abstract**: Attacks powered by Large Language Model (LLM) agents represent a growing threat to modern cybersecurity. To address this concern, we present LLM Honeypot, a system designed to monitor autonomous AI hacking agents. By augmenting a standard SSH honeypot with prompt injection and time-based analysis techniques, our framework aims to distinguish LLM agents among all attackers. Over a trial deployment of about three months in a public environment, we collected 8,130,731 hacking attempts and 8 potential AI agents. Our work demonstrates the emergence of AI-driven threats and their current level of usage, serving as an early warning of malicious LLM agents in the wild.



## **5. AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails**

cs.CR

7 pages, 3 figures, 2 tables, accepted in 4th IEEE International  Conference on AI in Cybersecurity (ICAIC)

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.03622v2) [paper-pdf](http://arxiv.org/pdf/2502.03622v2)

**Authors**: Rei Meguro, Ng S. T. Chong

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education.



## **6. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **7. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.



## **8. Preserving Privacy in Large Language Models: A Survey on Current Threats and Solutions**

cs.CR

Published in Transactions on Machine Learning Research (TMLR)  https://openreview.net/forum?id=Ss9MTTN7OL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.05212v2) [paper-pdf](http://arxiv.org/pdf/2408.05212v2)

**Authors**: Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sébastien Bratières, Emanuele Rodolà

**Abstract**: Large Language Models (LLMs) represent a significant advancement in artificial intelligence, finding applications across various domains. However, their reliance on massive internet-sourced datasets for training brings notable privacy issues, which are exacerbated in critical domains (e.g., healthcare). Moreover, certain application-specific scenarios may require fine-tuning these models on private data. This survey critically examines the privacy threats associated with LLMs, emphasizing the potential for these models to memorize and inadvertently reveal sensitive information. We explore current threats by reviewing privacy attacks on LLMs and propose comprehensive solutions for integrating privacy mechanisms throughout the entire learning pipeline. These solutions range from anonymizing training datasets to implementing differential privacy during training or inference and machine unlearning after training. Our comprehensive review of existing literature highlights ongoing challenges, available tools, and future directions for preserving privacy in LLMs. This work aims to guide the development of more secure and trustworthy AI systems by providing a thorough understanding of privacy preservation methods and their effectiveness in mitigating risks.



## **9. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18280v2) [paper-pdf](http://arxiv.org/pdf/2501.18280v2)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained significant attention recently, with various defense mechanisms developed to prevent harmful outputs, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the distribution of text embedding model outputs is significantly biased with a large mean. Inspired by this observation, we propose novel efficient methods to search for universal magic words that can attack text embedding models. The universal magic words as suffixes can move the embedding of any text towards the bias direction, therefore manipulate the similarity of any text pair and mislead safeguards. By appending magic words to user prompts and requiring LLMs to end answers with magic words, attackers can jailbreak the safeguard. To eradicate this security risk, we also propose defense mechanisms against such attacks, which can correct the biased distribution of text embeddings in a train-free manner.



## **10. Panza: Design and Analysis of a Fully-Local Personalized Text Writing Assistant**

cs.CL

Panza is available at https://github.com/IST-DASLab/PanzaMail

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2407.10994v4) [paper-pdf](http://arxiv.org/pdf/2407.10994v4)

**Authors**: Armand Nicolicioiu, Eugenia Iofinova, Andrej Jovanovic, Eldar Kurtic, Mahdi Nikdan, Andrei Panferov, Ilia Markov, Nir Shavit, Dan Alistarh

**Abstract**: The availability of powerful open-source large language models (LLMs) opens exciting use-cases, such as using personal data to fine-tune these models to imitate a user's unique writing style. Two key requirements for such assistants are personalization - in the sense that the assistant should recognizably reflect the user's own writing style - and privacy - users may justifiably be wary of uploading extremely personal data, such as their email archive, to a third-party service. In this paper, we present a new design and evaluation for such an automated assistant, for the specific use case of email generation, which we call Panza. Panza's personalization features are based on a combination of fine-tuning using a variant of the Reverse Instructions technique together with Retrieval-Augmented Generation (RAG). We demonstrate that this combination allows us to fine-tune an LLM to reflect a user's writing style using limited data, while executing on extremely limited resources, e.g. on a free Google Colab instance. Our key methodological contribution is the first detailed study of evaluation metrics for this personalized writing task, and of how different choices of system components--the use of RAG and of different fine-tuning approaches-impact the system's performance. Additionally, we demonstrate that very little data - under 100 email samples - are sufficient to create models that convincingly imitate humans. This finding showcases a previously-unknown attack vector in language models - that access to a small number of writing samples can allow a bad actor to cheaply create generative models that imitate a target's writing style. We are releasing the full Panza code as well as three new email datasets licensed for research use at https://github.com/IST-DASLab/PanzaMail.



## **11. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.



## **12. Confidence Elicitation: A New Attack Vector for Large Language Models**

cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.



## **13. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **14. Cyri: A Conversational AI-based Assistant for Supporting the Human User in Detecting and Responding to Phishing Attacks**

cs.HC

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05951v1) [paper-pdf](http://arxiv.org/pdf/2502.05951v1)

**Authors**: Antonio La Torre, Marco Angelini

**Abstract**: This work introduces Cyri, an AI-powered conversational assistant designed to support a human user in detecting and analyzing phishing emails by leveraging Large Language Models. Cyri has been designed to scrutinize emails for semantic features used in phishing attacks, such as urgency, and undesirable consequences, using an approach that unifies features already established in the literature with others by Cyri features extraction methodology. Cyri can be directly plugged into a client mail or webmail, ensuring seamless integration with the user's email workflow while maintaining data privacy through local processing. By performing analyses on the user's machine, Cyri eliminates the need to transmit sensitive email data over the internet, reducing associated security risks. The Cyri user interface has been designed to reduce habituation effects and enhance user engagement. It employs dynamic visual cues and context-specific explanations to keep users alert and informed while using emails. Additionally, it allows users to explore identified malicious semantic features both through conversation with the agent and visual exploration, obtaining the advantages of both modalities for expert or non-expert users. It also allows users to keep track of the conversation, supports the user in solving additional questions on both computed features or new parts of the mail, and applies its detection on demand. To evaluate Cyri, we crafted a comprehensive dataset of 420 phishing emails and 420 legitimate emails. Results demonstrate high effectiveness in identifying critical phishing semantic features fundamental to phishing detection. A user study involving 10 participants, both experts and non-experts, evaluated Cyri's effectiveness and usability. Results indicated that Cyri significantly aided users in identifying phishing emails and enhanced their understanding of phishing tactics.



## **15. Large Language Models are Easily Confused: A Quantitative Metric, Security Implications and Typological Analysis**

cs.CL

18 pages, 15 figures, 14 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.13237v2) [paper-pdf](http://arxiv.org/pdf/2410.13237v2)

**Authors**: Yiyi Chen, Qiongxiu Li, Russa Biswas, Johannes Bjerva

**Abstract**: Language Confusion is a phenomenon where Large Language Models (LLMs) generate text that is neither in the desired language, nor in a contextually appropriate language. This phenomenon presents a critical challenge in text generation by LLMs, often appearing as erratic and unpredictable behavior. We hypothesize that there are linguistic regularities to this inherent vulnerability in LLMs and shed light on patterns of language confusion across LLMs. We introduce a novel metric, Language Confusion Entropy, designed to directly measure and quantify this confusion, based on language distributions informed by linguistic typology and lexical variation. Comprehensive comparisons with the Language Confusion Benchmark (Marchisio et al., 2024) confirm the effectiveness of our metric, revealing patterns of language confusion across LLMs. We further link language confusion to LLM security, and find patterns in the case of multilingual embedding inversion attacks. Our analysis demonstrates that linguistic typology offers theoretically grounded interpretation, and valuable insights into leveraging language similarities as a prior for LLM alignment and security.



## **16. Arabic Dataset for LLM Safeguard Evaluation**

cs.CL

Accepted at NAACL 2025 Main Conference

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17040v2) [paper-pdf](http://arxiv.org/pdf/2410.17040v2)

**Authors**: Yasser Ashraf, Yuxia Wang, Bin Gu, Preslav Nakov, Timothy Baldwin

**Abstract**: The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.



## **17. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation**

cs.CR

This paper is accepted by conference WWW 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.20142v2) [paper-pdf](http://arxiv.org/pdf/2410.20142v2)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Retrieval-Augmented Generation (RAG) has been an effective approach to mitigate hallucinations in large language models (LLMs) by incorporating up-to-date and domain-specific knowledge. Recently, there has been a trend of storing up-to-date or copyrighted data in RAG knowledge databases instead of using it for LLM training. This practice has raised concerns about Membership Inference Attacks (MIAs), which aim to detect if a specific target document is stored in the RAG system's knowledge database so as to protect the rights of data producers. While research has focused on enhancing the trustworthiness of RAG systems, existing MIAs for RAG systems remain largely insufficient. Previous work either relies solely on the RAG system's judgment or is easily influenced by other documents or the LLM's internal knowledge, which is unreliable and lacks explainability. To address these limitations, we propose a Mask-Based Membership Inference Attacks (MBA) framework. Our framework first employs a masking algorithm that effectively masks a certain number of words in the target document. The masked text is then used to prompt the RAG system, and the RAG system is required to predict the mask values. If the target document appears in the knowledge database, the masked text will retrieve the complete target document as context, allowing for accurate mask prediction. Finally, we adopt a simple yet effective threshold-based method to infer the membership of target document by analyzing the accuracy of mask prediction. Our mask-based approach is more document-specific, making the RAG system's generation less susceptible to distractions from other documents or the LLM's internal knowledge. Extensive experiments demonstrate the effectiveness of our approach compared to existing baseline models.



## **18. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.



## **19. Dynamic Guided and Domain Applicable Safeguards for Enhanced Security in Large Language Models**

cs.AI

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2410.17922v2) [paper-pdf](http://arxiv.org/pdf/2410.17922v2)

**Authors**: Weidi Luo, He Cao, Zijing Liu, Yu Wang, Aidan Wong, Bing Feng, Yuan Yao, Yu Li

**Abstract**: With the extensive deployment of Large Language Models (LLMs), ensuring their safety has become increasingly critical. However, existing defense methods often struggle with two key issues: (i) inadequate defense capabilities, particularly in domain-specific scenarios like chemistry, where a lack of specialized knowledge can lead to the generation of harmful responses to malicious queries. (ii) over-defensiveness, which compromises the general utility and responsiveness of LLMs. To mitigate these issues, we introduce a multi-agents-based defense framework, Guide for Defense (G4D), which leverages accurate external information to provide an unbiased summary of user intentions and analytically grounded safety response guidance. Extensive experiments on popular jailbreak attacks and benign datasets show that our G4D can enhance LLM's robustness against jailbreak attacks on general and domain-specific scenarios without compromising the model's general functionality.



## **20. "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation**

cs.CR

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2409.02718v2) [paper-pdf](http://arxiv.org/pdf/2409.02718v2)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA.



## **21. HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor**

cs.LG

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2501.13677v2) [paper-pdf](http://arxiv.org/pdf/2501.13677v2)

**Authors**: Zihui Wu, Haichang Gao, Jiacheng Luo, Zhaoxiang Liu

**Abstract**: Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety.



## **22. Topic-Based Watermarks for Large Language Models**

cs.CR

Algorithms and new evaluations, 8 pages

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2404.02138v4) [paper-pdf](http://arxiv.org/pdf/2404.02138v4)

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday

**Abstract**: The indistinguishability of Large Language Model (LLM) output from human-authored content poses significant challenges, raising concerns about potential misuse of AI-generated text and its influence on future AI model training. Watermarking algorithms offer a viable solution by embedding detectable signatures into generated text. However, existing watermarking methods often entail trade-offs among attack robustness, generation quality, and additional overhead such as specialized frameworks or complex integrations. We propose a lightweight, topic-guided watermarking scheme for LLMs that partitions the vocabulary into topic-aligned token subsets. Given an input prompt, the scheme selects a relevant topic-specific token list, effectively "green-listing" semantically aligned tokens to embed robust marks while preserving the text's fluency and coherence. Experimental results across multiple LLMs and state-of-the-art benchmarks demonstrate that our method achieves comparable perplexity to industry-leading systems, including Google's SynthID-Text, yet enhances watermark robustness against paraphrasing and lexical perturbation attacks while introducing minimal performance overhead. Our approach avoids reliance on additional mechanisms beyond standard text generation pipelines, facilitating straightforward adoption, suggesting a practical path toward globally consistent watermarking of AI-generated content.



## **23. Watermarking Low-entropy Generation for Large Language Models: An Unbiased and Low-risk Method**

cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2405.14604v3) [paper-pdf](http://arxiv.org/pdf/2405.14604v3)

**Authors**: Minjia Mao, Dongjun Wei, Zeyu Chen, Xiao Fang, Michael Chau

**Abstract**: Recent advancements in large language models (LLMs) have highlighted the risk of misusing them, raising the need for accurate detection of LLM-generated content. In response, a viable solution is to inject imperceptible identifiers into LLMs, known as watermarks. Our research extends the existing watermarking methods by proposing the novel Sampling One Then Accepting (STA-1) method. STA-1 is an unbiased watermark that preserves the original token distribution in expectation and has a lower risk of producing unsatisfactory outputs in low-entropy scenarios compared to existing unbiased watermarks. In watermark detection, STA-1 does not require prompts or a white-box LLM, provides statistical guarantees, demonstrates high efficiency in detection time, and remains robust against various watermarking attacks. Experimental results on low-entropy and high-entropy datasets demonstrate that STA-1 achieves the above properties simultaneously, making it a desirable solution for watermarking LLMs. Implementation codes for this study are available online.



## **24. Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning**

cs.CL

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.07163v3) [paper-pdf](http://arxiv.org/pdf/2410.07163v3)

**Authors**: Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, Sijia Liu

**Abstract**: This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.



## **25. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.



## **26. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2411.13757v2) [paper-pdf](http://arxiv.org/pdf/2411.13757v2)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.



## **27. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.



## **28. Enhancing Phishing Email Identification with Large Language Models**

cs.CR

9 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04759v1) [paper-pdf](http://arxiv.org/pdf/2502.04759v1)

**Authors**: Catherine Lee

**Abstract**: Phishing has long been a common tactic used by cybercriminals and continues to pose a significant threat in today's digital world. When phishing attacks become more advanced and sophisticated, there is an increasing need for effective methods to detect and prevent them. To address the challenging problem of detecting phishing emails, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. In this work, we take steps to study the efficacy of large language models (LLMs) in detecting phishing emails. The experiments show that the LLM achieves a high accuracy rate at high precision; importantly, it also provides interpretable evidence for the decisions.



## **29. Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models**

cs.CR

Accepted by ICLR2025. url: https://openreview.net/forum?id=s20W12XTF8

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.02298v4) [paper-pdf](http://arxiv.org/pdf/2410.02298v4)

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications, ensuring both their safety and utility is paramount. Jailbreak attacks, which manipulate LLMs into generating harmful content, pose significant challenges to this balance. Existing defenses, such as prompt engineering and safety fine-tuning, often introduce computational overhead, increase inference latency, and lack runtime flexibility. Moreover, overly restrictive safety measures can degrade model utility by causing refusals of benign queries. In this paper, we introduce Jailbreak Antidote, a method that enables real-time adjustment of LLM safety preferences by manipulating a sparse subset of the model's internal states during inference. By shifting the model's hidden representations along a safety direction with varying strengths, we achieve flexible control over the safety-utility balance without additional token overhead or inference delays. Our analysis reveals that safety-related information in LLMs is sparsely distributed; adjusting approximately 5% of the internal state is as effective as modifying the entire state. Extensive experiments on nine LLMs (ranging from 2 billion to 72 billion parameters), evaluated against ten jailbreak attack methods and compared with six defense strategies, validate the effectiveness and efficiency of our approach. By directly manipulating internal states during reasoning, Jailbreak Antidote offers a lightweight, scalable solution that enhances LLM safety while preserving utility, opening new possibilities for real-time safety mechanisms in widely-deployed AI systems.



## **30. Membership Inference Attacks Against Vision-Language Models**

cs.CR

Accepted by USENIX'25; 22 pages, 28 figures;

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2501.18624v2) [paper-pdf](http://arxiv.org/pdf/2501.18624v2)

**Authors**: Yuke Hu, Zheng Li, Zhihao Liu, Yang Zhang, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Vision-Language Models (VLMs), built on pre-trained vision encoders and large language models (LLMs), have shown exceptional multi-modal understanding and dialog capabilities, positioning them as catalysts for the next technological revolution. However, while most VLM research focuses on enhancing multi-modal interaction, the risks of data misuse and leakage have been largely unexplored. This prompts the need for a comprehensive investigation of such risks in VLMs. In this paper, we conduct the first analysis of misuse and leakage detection in VLMs through the lens of membership inference attack (MIA). In specific, we focus on the instruction tuning data of VLMs, which is more likely to contain sensitive or unauthorized information. To address the limitation of existing MIA methods, we introduce a novel approach that infers membership based on a set of samples and their sensitivity to temperature, a unique parameter in VLMs. Based on this, we propose four membership inference methods, each tailored to different levels of background knowledge, ultimately arriving at the most challenging scenario. Our comprehensive evaluations show that these methods can accurately determine membership status, e.g., achieving an AUC greater than 0.8 targeting a small set consisting of only 5 samples on LLaVA.



## **31. Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Carrier Articles**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2408.11182v2) [paper-pdf](http://arxiv.org/pdf/2408.11182v2)

**Authors**: Zhilong Wang, Haizhou Wang, Nanqing Luo, Lan Zhang, Xiaoyan Sun, Yebo Cao, Peng Liu

**Abstract**: Large Language Model (LLM) jailbreak refers to a type of attack aimed to bypass the safeguard of an LLM to generate contents that are inconsistent with the safe usage guidelines. Based on the insights from the self-attention computation process, this paper proposes a novel blackbox jailbreak approach, which involves crafting the payload prompt by strategically injecting the prohibited query into a carrier article. The carrier article maintains the semantic proximity to the prohibited query, which is automatically produced by combining a hypernymy article and a context, both of which are generated from the prohibited query. The intuition behind the usage of carrier article is to activate the neurons in the model related to the semantics of the prohibited query while suppressing the neurons that will trigger the objectionable text. Carrier article itself is benign, and we leveraged prompt injection techniques to produce the payload prompt. We evaluate our approach using JailbreakBench, testing against four target models across 100 distinct jailbreak objectives. The experimental results demonstrate our method's superior effectiveness, achieving an average success rate of 63% across all target models, significantly outperforming existing blackbox jailbreak methods.



## **32. Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04322v1) [paper-pdf](http://arxiv.org/pdf/2502.04322v1)

**Authors**: Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, Marzyeh Ghassemi

**Abstract**: Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.



## **33. Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks**

cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04227v1) [paper-pdf](http://arxiv.org/pdf/2502.04227v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: We explore the feasibility and effectiveness of using LLM-driven autonomous systems for Assumed Breach penetration testing in enterprise networks. We introduce a novel prototype that, driven by Large Language Models (LLMs), can compromise accounts within a real-life Active Directory testbed. Our research provides a comprehensive evaluation of the prototype's capabilities, and highlights both strengths and limitations while executing attack. The evaluation uses a realistic simulation environment (Game of Active Directory, GOAD) to capture intricate interactions, stochastic outcomes, and timing dependencies that characterize live network scenarios. The study concludes that autonomous LLMs are able to conduct Assumed Breach simulations, potentially democratizing access to penetration testing for organizations facing budgetary constraints.   The prototype's source code, traces, and analyzed logs are released as open-source to enhance collective cybersecurity and facilitate future research in LLM-driven cybersecurity automation.



## **34. "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04204v1) [paper-pdf](http://arxiv.org/pdf/2502.04204v1)

**Authors**: Shaopeng Fu, Liang Ding, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the number of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix during jailbreaking to the length during AT. Our findings show that it is practical to defend "long-length" jailbreak attacks via efficient "short-length" AT. The code is available at https://github.com/fshp971/adv-icl.



## **35. Assessing and Prioritizing Ransomware Risk Based on Historical Victim Data**

cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.04421v1) [paper-pdf](http://arxiv.org/pdf/2502.04421v1)

**Authors**: Spencer Massengale, Philip Huff

**Abstract**: We present an approach to identifying which ransomware adversaries are most likely to target specific entities, thereby assisting these entities in formulating better protection strategies. Ransomware poses a formidable cybersecurity threat characterized by profit-driven motives, a complex underlying economy supporting criminal syndicates, and the overt nature of its attacks. This type of malware has consistently ranked among the most prevalent, with a rapid escalation in activity observed. Recent estimates indicate that approximately two-thirds of organizations experienced ransomware attacks in 2023 \cite{Sophos2023Ransomware}. A central tactic in ransomware campaigns is publicizing attacks to coerce victims into paying ransoms. Our study utilizes public disclosures from ransomware victims to predict the likelihood of an entity being targeted by a specific ransomware variant. We employ a Large Language Model (LLM) architecture that uses a unique chain-of-thought, multi-shot prompt methodology to define adversary SKRAM (Skills, Knowledge, Resources, Authorities, and Motivation) profiles from ransomware bulletins, threat reports, and news items. This analysis is enriched with publicly available victim data and is further enhanced by a heuristic for generating synthetic data that reflects victim profiles. Our work culminates in the development of a machine learning model that assists organizations in prioritizing ransomware threats and formulating defenses based on the tactics, techniques, and procedures (TTP) of the most likely attackers.



## **36. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.11782v3) [paper-pdf](http://arxiv.org/pdf/2410.11782v3)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Tianlong Chen, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **37. A Survey on Backdoor Threats in Large Language Models (LLMs): Attacks, Defenses, and Evaluations**

cs.CR

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2502.05224v1) [paper-pdf](http://arxiv.org/pdf/2502.05224v1)

**Authors**: Yihe Zhou, Tao Ni, Wei-Bin Lee, Qingchuan Zhao

**Abstract**: Large Language Models (LLMs) have achieved significantly advanced capabilities in understanding and generating human language text, which have gained increasing popularity over recent years. Apart from their state-of-the-art natural language processing (NLP) performance, considering their widespread usage in many industries, including medicine, finance, education, etc., security concerns over their usage grow simultaneously. In recent years, the evolution of backdoor attacks has progressed with the advancement of defense mechanisms against them and more well-developed features in the LLMs. In this paper, we adapt the general taxonomy for classifying machine learning attacks on one of the subdivisions - training-time white-box backdoor attacks. Besides systematically classifying attack methods, we also consider the corresponding defense methods against backdoor attacks. By providing an extensive summary of existing works, we hope this survey can serve as a guideline for inspiring future research that further extends the attack scenarios and creates a stronger defense against them for more robust LLMs.



## **38. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

Accepted at AAAI-25 Main Technical Track

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2408.06223v3) [paper-pdf](http://arxiv.org/pdf/2408.06223v3)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. We investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. We show that RMU unlearned models are robust against adversarial jailbreak attacks. Furthermore, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU--a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **39. GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models**

cs.CV

Code: https://github.com/jmiemirza/GLOV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2410.06154v5) [paper-pdf](http://arxiv.org/pdf/2410.06154v5)

**Authors**: M. Jehanzeb Mirza, Mengjie Zhao, Zhuoyuan Mao, Sivan Doveh, Wei Lin, Paul Gavrikov, Michael Dorkenwald, Shiqi Yang, Saurav Jha, Hiromi Wakaki, Yuki Mitsufuji, Horst Possegger, Rogerio Feris, Leonid Karlinsky, James Glass

**Abstract**: In this work, we propose GLOV, which enables Large Language Models (LLMs) to act as implicit optimizers for Vision-Language Models (VLMs) to enhance downstream vision tasks. GLOV prompts an LLM with the downstream task description, querying it for suitable VLM prompts (e.g., for zero-shot classification with CLIP). These prompts are ranked according to their fitness for the downstream vision task. In each respective optimization step, the ranked prompts are fed as in-context examples (with their accuracies) to equip the LLM with the knowledge of the type of prompts preferred by the downstream VLM. Furthermore, we explicitly guide the LLM's generation at each optimization step by adding an offset vector -- calculated from the embedding differences between previous positive and negative solutions -- to the intermediate layer of the network for the next generation. This offset vector biases the LLM generation toward the type of language the downstream VLM prefers, resulting in enhanced performance on the downstream vision tasks. We comprehensively evaluate our GLOV on two tasks: object recognition and the critical task of enhancing VLM safety. Our GLOV shows performance improvement by up to 15.0% and 57.5% for dual-encoder (e.g., CLIP) and encoder-decoder (e.g., LlaVA) models for object recognition and reduces the attack success rate (ASR) on state-of-the-art VLMs by up to $60.7\%$.



## **40. Aero-LLM: A Distributed Framework for Secure UAV Communication and Intelligent Decision-Making**

cs.CR

This manuscript was accepted by the 1st International Workshop on  Integrated Sensing, Communication, and Computing in Internet of Things (IoT)  Systems at the The 33rd International Conference on Computer Communications  and Networks (ICCCN 2024)

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.05220v1) [paper-pdf](http://arxiv.org/pdf/2502.05220v1)

**Authors**: Balakrishnan Dharmalingam, Rajdeep Mukherjee, Brett Piggott, Guohuan Feng, Anyi Liu

**Abstract**: Increased utilization of unmanned aerial vehicles (UAVs) in critical operations necessitates secure and reliable communication with Ground Control Stations (GCS). This paper introduces Aero-LLM, a framework integrating multiple Large Language Models (LLMs) to enhance UAV mission security and operational efficiency. Unlike conventional singular LLMs, Aero-LLM leverages multiple specialized LLMs for various tasks, such as inferencing, anomaly detection, and forecasting, deployed across onboard systems, edge, and cloud servers. This dynamic, distributed architecture reduces performance bottleneck and increases security capabilities. Aero-LLM's evaluation demonstrates outstanding task-specific metrics and robust defense against cyber threats, significantly enhancing UAV decision-making and operational capabilities and security resilience against cyber attacks, setting a new standard for secure, intelligent UAV operations.



## **41. Exploring the Security Threats of Knowledge Base Poisoning in Retrieval-Augmented Code Generation**

cs.CR

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03233v1) [paper-pdf](http://arxiv.org/pdf/2502.03233v1)

**Authors**: Bo Lin, Shangwen Wang, Liqian Chen, Xiaoguang Mao

**Abstract**: The integration of Large Language Models (LLMs) into software development has revolutionized the field, particularly through the use of Retrieval-Augmented Code Generation (RACG) systems that enhance code generation with information from external knowledge bases. However, the security implications of RACG systems, particularly the risks posed by vulnerable code examples in the knowledge base, remain largely unexplored. This risk is particularly concerning given that public code repositories, which often serve as the sources for knowledge base collection in RACG systems, are usually accessible to anyone in the community. Malicious attackers can exploit this accessibility to inject vulnerable code into the knowledge base, making it toxic. Once these poisoned samples are retrieved and incorporated into the generated code, they can propagate security vulnerabilities into the final product. This paper presents the first comprehensive study on the security risks associated with RACG systems, focusing on how vulnerable code in the knowledge base compromises the security of generated code. We investigate the LLM-generated code security across different settings through extensive experiments using four major LLMs, two retrievers, and two poisoning scenarios. Our findings highlight the significant threat of knowledge base poisoning, where even a single poisoned code example can compromise up to 48% of generated code. Our findings provide crucial insights into vulnerability introduction in RACG systems and offer practical mitigation recommendations, thereby helping improve the security of LLM-generated code in future works.



## **42. ImgTrojan: Jailbreaking Vision-Language Models with ONE Image**

cs.CV

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2403.02910v3) [paper-pdf](http://arxiv.org/pdf/2403.02910v3)

**Authors**: Xijia Tao, Shuai Zhong, Lei Li, Qi Liu, Lingpeng Kong

**Abstract**: There has been an increasing interest in the alignment of large language models (LLMs) with human values. However, the safety issues of their integration with a vision module, or vision language models (VLMs), remain relatively underexplored. In this paper, we propose a novel jailbreaking attack against VLMs, aiming to bypass their safety barrier when a user inputs harmful instructions. A scenario where our poisoned (image, text) data pairs are included in the training data is assumed. By replacing the original textual captions with malicious jailbreak prompts, our method can perform jailbreak attacks with the poisoned images. Moreover, we analyze the effect of poison ratios and positions of trainable parameters on our attack's success rate. For evaluation, we design two metrics to quantify the success rate and the stealthiness of our attack. Together with a list of curated harmful instructions, a benchmark for measuring attack efficacy is provided. We demonstrate the efficacy of our attack by comparing it with baseline methods.



## **43. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.03052v1) [paper-pdf](http://arxiv.org/pdf/2502.03052v1)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.



## **44. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

cs.CR

Accepted by USENIX Security Symposium 2025. Please cite the  conference version of this paper, i.e., "Xunguang Wang, Daoyuan Wu, Zhenlan  Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, and  Juergen Rahmel. SelfDefend: LLMs Can Defend Themselves against Jailbreaking  in a Practical Manner. In Proc. USENIX Security, 2025."

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2406.05498v3) [paper-pdf](http://arxiv.org/pdf/2406.05498v3)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into multiple categories: human-based, optimization-based, generation-based, and the recent indirect and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delays to user prompts, as well as be compatible with both open-source and closed-source LLMs. Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM as a defense instance (in detection state) to concurrently protect the target LLM instance (in normal answering state) in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs can identify harmful prompts or intentions in user queries, which we empirically validate using mainstream GPT-3.5/4 models against major jailbreak attacks. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. When deployed to protect GPT-3.5/4, Claude, Llama-2-7b/13b, and Mistral, these models outperform seven state-of-the-art defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. Further experiments show that the tuned models are robust to adaptive jailbreaks and prompt injections.



## **45. Lost in Overlap: Exploring Logit-based Watermark Collision in LLMs**

cs.CL

Long Paper, 9 pages, accepted at NAACL 2025 Findings

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2403.10020v3) [paper-pdf](http://arxiv.org/pdf/2403.10020v3)

**Authors**: Yiyang Luo, Ke Lin, Chao Gu, Jiahui Hou, Lijie Wen, Ping Luo

**Abstract**: The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.



## **46. Large Language Model Adversarial Landscape Through the Lens of Attack Objectives**

cs.CR

15 pages

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02960v1) [paper-pdf](http://arxiv.org/pdf/2502.02960v1)

**Authors**: Nan Wang, Kane Walter, Yansong Gao, Alsharif Abuadbba

**Abstract**: Large Language Models (LLMs) represent a transformative leap in artificial intelligence, enabling the comprehension, generation, and nuanced interaction with human language on an unparalleled scale. However, LLMs are increasingly vulnerable to a range of adversarial attacks that threaten their privacy, reliability, security, and trustworthiness. These attacks can distort outputs, inject biases, leak sensitive information, or disrupt the normal functioning of LLMs, posing significant challenges across various applications.   In this paper, we provide a novel comprehensive analysis of the adversarial landscape of LLMs, framed through the lens of attack objectives. By concentrating on the core goals of adversarial actors, we offer a fresh perspective that examines threats from the angles of privacy, integrity, availability, and misuse, moving beyond conventional taxonomies that focus solely on attack techniques. This objective-driven adversarial landscape not only highlights the strategic intent behind different adversarial approaches but also sheds light on the evolving nature of these threats and the effectiveness of current defenses. Our analysis aims to guide researchers and practitioners in better understanding, anticipating, and mitigating these attacks, ultimately contributing to the development of more resilient and robust LLM systems.



## **47. How Much Do Code Language Models Remember? An Investigation on Data Extraction Attacks before and after Fine-tuning**

cs.CR

MSR 2025

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2501.17501v2) [paper-pdf](http://arxiv.org/pdf/2501.17501v2)

**Authors**: Fabio Salerno, Ali Al-Kaswan, Maliheh Izadi

**Abstract**: Code language models, while widely popular, are often trained on unsanitized source code gathered from across the Internet. Previous work revealed that pre-trained models can remember the content of their training data and regurgitate them through data extraction attacks. Due to the large size of current models, only a few entities have the resources for pre-training such models. However, fine-tuning requires fewer resources and is increasingly used by both small and large entities for its effectiveness on specialized data. Such small curated data for fine-tuning might contain sensitive information or proprietary assets. In this study, we attack both pre-trained and fine-tuned code language models to investigate the extent of data extractability. We first develop a custom benchmark to assess the vulnerability of both pre-training and fine-tuning samples to extraction attacks. Our findings reveal that 54.9% of extractable pre-training data could be retrieved from StarCoder2-15B, whereas this number decreased to 23.5% after fine-tuning. This indicates that fine-tuning reduces the extractability of pre-training data. However, compared to larger models, fine-tuning smaller models increases their vulnerability to data extraction attacks on fine-tuning data. Given the potential sensitivity of fine-tuning data, this can lead to more severe consequences. Lastly, we also manually analyzed 2000 extractable samples before and after fine-tuning. We also found that data carriers and licensing information are the most likely data categories to be memorized from pre-trained and fine-tuned models, while the latter is the most likely to be forgotten after fine-tuning.



## **48. SimMark: A Robust Sentence-Level Similarity-Based Watermarking Algorithm for Large Language Models**

cs.CL

15 pages, 5 tables, 6 figures

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.02787v1) [paper-pdf](http://arxiv.org/pdf/2502.02787v1)

**Authors**: Amirhossein Dabiriaghdam, Lele Wang

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for reliable methods to detect whether a text is generated by such models. In this paper, we propose SimMark, a posthoc watermarking algorithm that makes LLMs' outputs traceable without requiring access to the model's internal logits, enabling compatibility with a wide range of LLMs, including API-only models. By leveraging the similarity of semantic sentence embeddings and rejection sampling to impose detectable statistical patterns imperceptible to humans, and employing a soft counting mechanism, SimMark achieves robustness against paraphrasing attacks. Experimental results demonstrate that SimMark sets a new benchmark for robust watermarking of LLM-generated content, surpassing prior sentence-level watermarking techniques in robustness, sampling efficiency, and applicability across diverse domains, all while preserving the text quality.



## **49. MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction**

cs.CL

**SubmitDate**: 2025-02-05    [abs](http://arxiv.org/abs/2502.04360v1) [paper-pdf](http://arxiv.org/pdf/2502.04360v1)

**Authors**: Xiao Hu, Eric Liu, Weizhou Wang, Xiangyu Guo, David Lie

**Abstract**: Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowledge retrieved from external sources. The use of private resources and data in constructing these external data stores can expose them to risks of extraction attacks, in which attackers attempt to steal data from these private databases. Existing RAG extraction attacks often rely on manually crafted prompts, which limit their effectiveness. In this paper, we introduce a framework called MARAGE for optimizing an adversarial string that, when appended to user queries submitted to a target RAG system, causes outputs containing the retrieved RAG data verbatim. MARAGE leverages a continuous optimization scheme that integrates gradients from multiple models with different architectures simultaneously to enhance the transferability of the optimized string to unseen models. Additionally, we propose a strategy that emphasizes the initial tokens in the target RAG data, further improving the attack's generalizability. Evaluations show that MARAGE consistently outperforms both manual and optimization-based baselines across multiple LLMs and RAG datasets, while maintaining robust transferability to previously unseen models. Moreover, we conduct probing tasks to shed light on the reasons why MARAGE is more effective compared to the baselines and to analyze the impact of our approach on the model's internal state.



## **50. Certifying LLM Safety against Adversarial Prompting**

cs.CL

Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq

**SubmitDate**: 2025-02-04    [abs](http://arxiv.org/abs/2309.02705v4) [paper-pdf](http://arxiv.org/pdf/2309.02705v4)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



