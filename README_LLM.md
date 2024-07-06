# Latest Large Language Model Attack Papers
**update at 2024-07-06 15:43:41**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge**

cs.CL

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2404.05880v2) [paper-pdf](http://arxiv.org/pdf/2404.05880v2)

**Authors**: Weikai Lu, Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Zelin Chen, Huiping Zhuang, Cen Chen

**Abstract**: Jailbreaking attacks can enable Large Language Models (LLMs) to bypass the safeguard and generate harmful content. Existing jailbreaking defense methods have failed to address the fundamental issue that harmful knowledge resides within the model, leading to potential jailbreak risks for LLMs. In this paper, we propose a novel defense method called Eraser, which mainly includes three goals: unlearning harmful knowledge, retaining general knowledge, and maintaining safety alignment. The intuition is that if an LLM forgets the specific knowledge required to answer a harmful question, it will no longer have the ability to answer harmful questions. The training of Erase does not actually require the model's own harmful knowledge, and it can benefit from unlearning general answers related to harmful queries, which means it does not need assistance from the red team. The experimental results show that Eraser can significantly reduce the jailbreaking success rate for various attacks without compromising the general capabilities of the model. Our codes are available at https://github.com/ZeroNLP/Eraser.



## **2. SOS! Soft Prompt Attack Against Open-Source Large Language Models**

cs.CR

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03160v1) [paper-pdf](http://arxiv.org/pdf/2407.03160v1)

**Authors**: Ziqing Yang, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Open-source large language models (LLMs) have become increasingly popular among both the general public and industry, as they can be customized, fine-tuned, and freely used. However, some open-source LLMs require approval before usage, which has led to third parties publishing their own easily accessible versions. Similarly, third parties have been publishing fine-tuned or quantized variants of these LLMs. These versions are particularly appealing to users because of their ease of access and reduced computational resource demands. This trend has increased the risk of training time attacks, compromising the integrity and security of LLMs. In this work, we present a new training time attack, SOS, which is designed to be low in computational demand and does not require clean data or modification of the model weights, thereby maintaining the model's utility intact. The attack addresses security issues in various scenarios, including the backdoor attack, jailbreak attack, and prompt stealing attack. Our experimental findings demonstrate that the proposed attack is effective across all evaluated targets. Furthermore, we present the other side of our SOS technique, namely the copyright token -- a novel technique that enables users to mark their copyrighted content and prevent models from using it.



## **3. JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets**

cs.HC

18 pages, 9 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03045v1) [paper-pdf](http://arxiv.org/pdf/2407.03045v1)

**Authors**: Zhihua Jin, Shiyi Liu, Haotian Li, Xun Zhao, Huamin Qu

**Abstract**: Large Language Models (LLMs) have gained significant attention but also raised concerns due to the risk of misuse. Jailbreak prompts, a popular type of adversarial attack towards LLMs, have appeared and constantly evolved to breach the safety protocols of LLMs. To address this issue, LLMs are regularly updated with safety patches based on reported jailbreak prompts. However, malicious users often keep their successful jailbreak prompts private to exploit LLMs. To uncover these private jailbreak prompts, extensive analysis of large-scale conversational datasets is necessary to identify prompts that still manage to bypass the system's defenses. This task is highly challenging due to the immense volume of conversation data, diverse characteristics of jailbreak prompts, and their presence in complex multi-turn conversations. To tackle these challenges, we introduce JailbreakHunter, a visual analytics approach for identifying jailbreak prompts in large-scale human-LLM conversational datasets. We have designed a workflow with three analysis levels: group-level, conversation-level, and turn-level. Group-level analysis enables users to grasp the distribution of conversations and identify suspicious conversations using multiple criteria, such as similarity with reported jailbreak prompts in previous research and attack success rates. Conversation-level analysis facilitates the understanding of the progress of conversations and helps discover jailbreak prompts within their conversation contexts. Turn-level analysis allows users to explore the semantic similarity and token overlap between a singleturn prompt and the reported jailbreak prompts, aiding in the identification of new jailbreak strategies. The effectiveness and usability of the system were verified through multiple case studies and expert interviews.



## **4. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

cs.CR

To be presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02596v1) [paper-pdf](http://arxiv.org/pdf/2407.02596v1)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing large parts of their training data, making them vulnerable to extraction attacks. Existing research on these attacks remains limited in scope, often studying isolated trends rather than the real-world interactions with these models. In this paper, we revisit extraction attacks from an adversarial perspective, exploiting the brittleness of language models. We find significant churn in extraction attack trends, i.e., even minor, unintuitive changes to the prompt, or targeting smaller models and older checkpoints, can exacerbate the risks of extraction by up to $2-4 \times$. Moreover, relying solely on the widely accepted verbatim match underestimates the extent of extracted information, and we provide various alternatives to more accurately capture the true risks of extraction. We conclude our discussion with data deduplication, a commonly suggested mitigation strategy, and find that while it addresses some memorization concerns, it remains vulnerable to the same escalation of extraction risks against a real-world adversary. Our findings highlight the necessity of acknowledging an adversary's true capabilities to avoid underestimating extraction risks.



## **5. A False Sense of Safety: Unsafe Information Leakage in 'Safe' AI Responses**

cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02551v1) [paper-pdf](http://arxiv.org/pdf/2407.02551v1)

**Authors**: David Glukhov, Ziwen Han, Ilia Shumailov, Vardan Papyan, Nicolas Papernot

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreaks$\unicode{x2013}$methods to elicit harmful or generally impermissible outputs. Safety measures are developed and assessed on their effectiveness at defending against jailbreak attacks, indicating a belief that safety is equivalent to robustness. We assert that current defense mechanisms, such as output filters and alignment fine-tuning, are, and will remain, fundamentally insufficient for ensuring model safety. These defenses fail to address risks arising from dual-intent queries and the ability to composite innocuous outputs to achieve harmful goals. To address this critical gap, we introduce an information-theoretic threat model called inferential adversaries who exploit impermissible information leakage from model outputs to achieve malicious goals. We distinguish these from commonly studied security adversaries who only seek to force victim models to generate specific impermissible outputs. We demonstrate the feasibility of automating inferential adversaries through question decomposition and response aggregation. To provide safety guarantees, we define an information censorship criterion for censorship mechanisms, bounding the leakage of impermissible information. We propose a defense mechanism which ensures this bound and reveal an intrinsic safety-utility trade-off. Our work provides the first theoretically grounded understanding of the requirements for releasing safe LLMs and the utility costs involved.



## **6. Uncovering Safety Risks of Large Language Models through Concept Activation Vector**

cs.CL

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2404.12038v3) [paper-pdf](http://arxiv.org/pdf/2404.12038v3)

**Authors**: Zhihao Xu, Ruixuan Huang, Changyu Chen, Shuai Wang, Xiting Wang

**Abstract**: Despite careful safety alignment, current large language models (LLMs) remain vulnerable to various attacks. To further unveil the safety risks of LLMs, we introduce a Safety Concept Activation Vector (SCAV) framework, which effectively guides the attacks by accurately interpreting LLMs' safety mechanisms. We then develop an SCAV-guided attack method that can generate both attack prompts and embedding-level attacks with automatically selected perturbation hyperparameters. Both automatic and human evaluations demonstrate that our attack method significantly improves the attack success rate and response quality while requiring less training data. Additionally, we find that our generated attack prompts may be transferable to GPT-4, and the embedding-level attacks may also be transferred to other white-box LLMs whose parameters are known. Our experiments further uncover the safety risks present in current LLMs. For example, we find that six out of seven open-source LLMs that we attack consistently provide relevant answers to more than 85\% malicious instructions. Finally, we provide insights into the safety mechanism of LLMs.



## **7. Adversarial Search Engine Optimization for Large Language Models**

cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2406.18382v2) [paper-pdf](http://arxiv.org/pdf/2406.18382v2)

**Authors**: Fredrik Nestaas, Edoardo Debenedetti, Florian Tramèr

**Abstract**: Large Language Models (LLMs) are increasingly used in applications where the model selects from competing third-party content, such as in LLM-powered search engines or chatbot plugins. In this paper, we introduce Preference Manipulation Attacks, a new class of attacks that manipulate an LLM's selections to favor the attacker. We demonstrate that carefully crafted website content or plugin documentations can trick an LLM to promote the attacker products and discredit competitors, thereby increasing user traffic and monetization. We show this leads to a prisoner's dilemma, where all parties are incentivized to launch attacks, but the collective effect degrades the LLM's outputs for everyone. We demonstrate our attacks on production LLM search engines (Bing and Perplexity) and plugin APIs (for GPT-4 and Claude). As LLMs are increasingly used to rank third-party content, we expect Preference Manipulation Attacks to emerge as a significant threat.



## **8. SoP: Unlock the Power of Social Facilitation for Automatic Jailbreak Attack**

cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01902v1) [paper-pdf](http://arxiv.org/pdf/2407.01902v1)

**Authors**: Yan Yang, Zeguan Xiao, Xin Lu, Hongru Wang, Hailiang Huang, Guanhua Chen, Yun Chen

**Abstract**: The widespread applications of large language models (LLMs) have brought about concerns regarding their potential misuse. Although aligned with human preference data before release, LLMs remain vulnerable to various malicious attacks. In this paper, we adopt a red-teaming strategy to enhance LLM safety and introduce SoP, a simple yet effective framework to design jailbreak prompts automatically. Inspired by the social facilitation concept, SoP generates and optimizes multiple jailbreak characters to bypass the guardrails of the target LLM. Different from previous work which relies on proprietary LLMs or seed jailbreak templates crafted by human expertise, SoP can generate and optimize the jailbreak prompt in a cold-start scenario using open-sourced LLMs without any seed jailbreak templates. Experimental results show that SoP achieves attack success rates of 88% and 60% in bypassing the safety alignment of GPT-3.5-1106 and GPT-4, respectively. Furthermore, we extensively evaluate the transferability of the generated templates across different LLMs and held-out malicious requests, while also exploring defense strategies against the jailbreak attack designed by SoP. Code is available at https://github.com/Yang-Yan-Yang-Yan/SoP.



## **9. Revisiting Backdoor Attacks against Large Vision-Language Models**

cs.CV

24 pages, 8 figures

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2406.18844v3) [paper-pdf](http://arxiv.org/pdf/2406.18844v3)

**Authors**: Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Ee-Chien Chang, Xiaochun Cao

**Abstract**: Instruction tuning enhances large vision-language models (LVLMs) but raises security risks through potential backdoor attacks due to their openness. Previous backdoor studies focus on enclosed scenarios with consistent training and testing instructions, neglecting the practical domain gaps that could affect attack effectiveness. This paper empirically examines the generalizability of backdoor attacks during the instruction tuning of LVLMs for the first time, revealing certain limitations of most backdoor strategies in practical scenarios. We quantitatively evaluate the generalizability of six typical backdoor attacks on image caption benchmarks across multiple LVLMs, considering both visual and textual domain offsets. Our findings indicate that attack generalizability is positively correlated with the backdoor trigger's irrelevance to specific images/models and the preferential correlation of the trigger pattern. Additionally, we modify existing backdoor attacks based on the above key observations, demonstrating significant improvements in cross-domain scenario generalizability (+86% attack success rate). Notably, even without access to the instruction datasets, a multimodal instruction set can be successfully poisoned with a very low poisoning rate (0.2%), achieving an attack success rate of over 97%. This paper underscores that even simple traditional backdoor strategies pose a serious threat to LVLMs, necessitating more attention and in-depth research.



## **10. Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything**

cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.02534v1) [paper-pdf](http://arxiv.org/pdf/2407.02534v1)

**Authors**: Xiaotian Zou, Yongkang Chen

**Abstract**: Large Visual Language Models (VLMs) such as GPT-4 have achieved remarkable success in generating comprehensive and nuanced responses, surpassing the capabilities of large language models. However, with the integration of visual inputs, new security concerns emerge, as malicious attackers can exploit multiple modalities to achieve their objectives. This has led to increasing attention on the vulnerabilities of VLMs to jailbreak. Most existing research focuses on generating adversarial images or nonsensical image collections to compromise these models. However, the challenge of leveraging meaningful images to produce targeted textual content using the VLMs' logical comprehension of images remains unexplored. In this paper, we explore the problem of logical jailbreak from meaningful images to text. To investigate this issue, we introduce a novel dataset designed to evaluate flowchart image jailbreak. Furthermore, we develop a framework for text-to-text jailbreak using VLMs. Finally, we conduct an extensive evaluation of the framework on GPT-4o and GPT-4-vision-preview, with jailbreak rates of 92.8% and 70.0%, respectively. Our research reveals significant vulnerabilities in current VLMs concerning image-to-text jailbreak. These findings underscore the need for a deeper examination of the security flaws in VLMs before their practical deployment.



## **11. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .



## **12. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.



## **13. A Fingerprint for Large Language Models**

cs.CR

https://scholar.google.com/citations?user=IdiF7M0AAAAJ&hl=en

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01235v1) [paper-pdf](http://arxiv.org/pdf/2407.01235v1)

**Authors**: Zhiguang Yang, Hanzhou Wu

**Abstract**: Recent advances show that scaling a pre-trained language model could achieve state-of-the-art performance on many downstream tasks, prompting large language models (LLMs) to become a hot research topic in the field of artificial intelligence. However, due to the resource-intensive nature of training LLMs from scratch, it is urgent and crucial to protect the intellectual property of LLMs against infringement. This has motivated the authors in this paper to propose a novel black-box fingerprinting technique for LLMs, which requires neither model training nor model fine-tuning. We first demonstrate that the outputs of LLMs span a unique vector space associated with each model. We model the problem of ownership authentication as the task of evaluating the similarity between the victim model's space and the output's space of the suspect model. To deal with this problem, we propose two solutions, where the first solution involves verifying whether the outputs of the suspected large model are in the same space as those of the victim model, enabling rapid identification of model infringement, and the second one reconstructs the union of the vector spaces for LLM outputs and the victim model to address situations where the victim model has undergone the Parameter-Efficient Fine-Tuning (PEFT) attacks. Experimental results indicate that the proposed technique achieves superior performance in ownership verification and robustness against PEFT attacks. This work reveals inherent characteristics of LLMs and provides a promising solution for ownership verification of LLMs in black-box scenarios, ensuring efficiency, generality and practicality.



## **14. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications**

cs.LG

22 pages, 9 figures. Project page is available at  https://boyiwei.com/alignment-attribution/

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2402.05162v3) [paper-pdf](http://arxiv.org/pdf/2402.05162v3)

**Authors**: Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, Peter Henderson

**Abstract**: Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3\%$ at the parameter level and $2.5\%$ at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model's safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.



## **15. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00869v1) [paper-pdf](http://arxiv.org/pdf/2407.00869v1)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.



## **16. Virtual Context: Enhancing Jailbreak Attacks with Special Token Injection**

cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19845v1) [paper-pdf](http://arxiv.org/pdf/2406.19845v1)

**Authors**: Yuqi Zhou, Lin Lu, Hanchi Sun, Pan Zhou, Lichao Sun

**Abstract**: Jailbreak attacks on large language models (LLMs) involve inducing these models to generate harmful content that violates ethics or laws, posing a significant threat to LLM security. Current jailbreak attacks face two main challenges: low success rates due to defensive measures and high resource requirements for crafting specific prompts. This paper introduces Virtual Context, which leverages special tokens, previously overlooked in LLM security, to improve jailbreak attacks. Virtual Context addresses these challenges by significantly increasing the success rates of existing jailbreak methods and requiring minimal background knowledge about the target model, thus enhancing effectiveness in black-box settings without additional overhead. Comprehensive evaluations show that Virtual Context-assisted jailbreak attacks can improve the success rates of four widely used jailbreak methods by approximately 40% across various LLMs. Additionally, applying Virtual Context to original malicious behaviors still achieves a notable jailbreak effect. In summary, our research highlights the potential of special tokens in jailbreak attacks and recommends including this threat in red-teaming testing to comprehensively enhance LLM security.



## **17. SafeAligner: Safety Alignment against Jailbreak Attacks via Response Disparity Guidance**

cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.18118v2) [paper-pdf](http://arxiv.org/pdf/2406.18118v2)

**Authors**: Caishuang Huang, Wanxu Zhao, Rui Zheng, Huijie Lv, Shihan Dou, Sixian Li, Xiao Wang, Enyu Zhou, Junjie Ye, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: As the development of large language models (LLMs) rapidly advances, securing these models effectively without compromising their utility has become a pivotal area of research. However, current defense strategies against jailbreak attacks (i.e., efforts to bypass security protocols) often suffer from limited adaptability, restricted general capability, and high cost. To address these challenges, we introduce SafeAligner, a methodology implemented at the decoding stage to fortify defenses against jailbreak attacks. We begin by developing two specialized models: the Sentinel Model, which is trained to foster safety, and the Intruder Model, designed to generate riskier responses. SafeAligner leverages the disparity in security levels between the responses from these models to differentiate between harmful and beneficial tokens, effectively guiding the safety alignment by altering the output token distribution of the target model. Extensive experiments show that SafeAligner can increase the likelihood of beneficial tokens, while reducing the occurrence of harmful ones, thereby ensuring secure alignment with minimal loss to generality.



## **18. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

cs.AI

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2309.10253v4) [paper-pdf](http://arxiv.org/pdf/2309.10253v4)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.



## **19. Seeing Is Believing: Black-Box Membership Inference Attacks Against Retrieval Augmented Generation**

cs.CR

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19234v1) [paper-pdf](http://arxiv.org/pdf/2406.19234v1)

**Authors**: Yuying Li, Gaoyang Liu, Yang Yang, Chen Wang

**Abstract**: Retrieval-Augmented Generation (RAG) is a state-of-the-art technique that enhances Large Language Models (LLMs) by retrieving relevant knowledge from an external, non-parametric database. This approach aims to mitigate common LLM issues such as hallucinations and outdated knowledge. Although existing research has demonstrated security and privacy vulnerabilities within RAG systems, making them susceptible to attacks like jailbreaks and prompt injections, the security of the RAG system's external databases remains largely underexplored. In this paper, we employ Membership Inference Attacks (MIA) to determine whether a sample is part of the knowledge database of a RAG system, using only black-box API access. Our core hypothesis posits that if a sample is a member, it will exhibit significant similarity to the text generated by the RAG system. To test this, we compute the cosine similarity and the model's perplexity to establish a membership score, thereby building robust features. We then introduce two novel attack strategies: a Threshold-based Attack and a Machine Learning-based Attack, designed to accurately identify membership. Experimental validation of our methods has achieved a ROC AUC of 82%.



## **20. Chat AI: A Seamless Slurm-Native Solution for HPC-Based Services**

cs.DC

27 pages, 5 figures, 2 tables

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2407.00110v1) [paper-pdf](http://arxiv.org/pdf/2407.00110v1)

**Authors**: Ali Doosthosseini, Jonathan Decker, Hendrik Nolte, Julian M. Kunkel

**Abstract**: The increasing adoption of large language models (LLMs) has created a pressing need for an efficient, secure and private serving infrastructure, which allows researchers to run open-source or custom fine-tuned LLMs and ensures users that their data remains private and is not stored without their consent. While high-performance computing (HPC) systems equipped with state-of-the-art GPUs are well-suited for training LLMs, their batch scheduling paradigm is not designed to support real-time serving of AI applications. Cloud systems, on the other hand, are well suited for web services but commonly lack access to the computational power of clusters, especially expensive and scarce high-end GPUs, which are required for optimal inference speed. We propose an architecture with an implementation consisting of a web service that runs on a cloud VM with secure access to a scalable backend running a multitude of AI models on HPC systems. By offering a web service using our HPC infrastructure to host LLMs, we leverage the trusted environment of local universities and research centers to offer a private and secure alternative to commercial LLM services. Our solution natively integrates with Slurm, enabling seamless deployment on HPC clusters and is able to run side by side with regular Slurm workloads, while utilizing gaps in the schedule created by Slurm. In order to ensure the security of the HPC system, we use the SSH ForceCommand directive to construct a robust circuit breaker, which prevents successful attacks on the web-facing server from affecting the cluster. We have successfully deployed our system as a production service, and made the source code available at https://github.com/gwdg/chat-ai



## **21. Assessing the Effectiveness of LLMs in Android Application Vulnerability Analysis**

cs.CR

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18894v1) [paper-pdf](http://arxiv.org/pdf/2406.18894v1)

**Authors**: Vasileios Kouliaridis, Georgios Karopoulos, Georgios Kambourakis

**Abstract**: The increasing frequency of attacks on Android applications coupled with the recent popularity of large language models (LLMs) necessitates a comprehensive understanding of the capabilities of the latter in identifying potential vulnerabilities, which is key to mitigate the overall risk. To this end, the work at hand compares the ability of nine state-of-the-art LLMs to detect Android code vulnerabilities listed in the latest Open Worldwide Application Security Project (OWASP) Mobile Top 10. Each LLM was evaluated against an open dataset of over 100 vulnerable code samples, including obfuscated ones, assessing each model's ability to identify key vulnerabilities. Our analysis reveals the strengths and weaknesses of each LLM, identifying important factors that contribute to their performance. Additionally, we offer insights into context augmentation with retrieval-augmented generation (RAG) for detecting Android code vulnerabilities, which in turn may propel secure application development. Finally, while the reported findings regarding code vulnerability analysis show promise, they also reveal significant discrepancies among the different LLMs.



## **22. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18849v1) [paper-pdf](http://arxiv.org/pdf/2406.18849v1)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.



## **23. Jailbreaking LLMs with Arabic Transliteration and Arabizi**

cs.LG

14 pages, 4 figures

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18725v1) [paper-pdf](http://arxiv.org/pdf/2406.18725v1)

**Authors**: Mansour Al Ghanim, Saleh Almohaimeed, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: This study identifies the potential vulnerabilities of Large Language Models (LLMs) to 'jailbreak' attacks, specifically focusing on the Arabic language and its various forms. While most research has concentrated on English-based prompt manipulation, our investigation broadens the scope to investigate the Arabic language. We initially tested the AdvBench benchmark in Standardized Arabic, finding that even with prompt manipulation techniques like prefix injection, it was insufficient to provoke LLMs into generating unsafe content. However, when using Arabic transliteration and chatspeak (or arabizi), we found that unsafe content could be produced on platforms like OpenAI GPT-4 and Anthropic Claude 3 Sonnet. Our findings suggest that using Arabic and its various forms could expose information that might remain hidden, potentially increasing the risk of jailbreak attacks. We hypothesize that this exposure could be due to the model's learned connection to specific words, highlighting the need for more comprehensive safety training across all language forms.



## **24. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.



## **25. Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.10794v2) [paper-pdf](http://arxiv.org/pdf/2406.10794v2)

**Authors**: Yuping Lin, Pengfei He, Han Xu, Yue Xing, Makoto Yamada, Hui Liu, Jiliang Tang

**Abstract**: Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.



## **26. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.



## **27. Enhancing Data Privacy in Large Language Models through Private Association Editing**

cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18221v1) [paper-pdf](http://arxiv.org/pdf/2406.18221v1)

**Authors**: Davide Venditti, Elena Sofia Ruzzetti, Giancarlo A. Xompero, Cristina Giannone, Andrea Favalli, Raniero Romagnoli, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) are powerful tools with extensive applications, but their tendency to memorize private information raises significant concerns as private data leakage can easily happen. In this paper, we introduce Private Association Editing (PAE), a novel defense approach for private data leakage. PAE is designed to effectively remove Personally Identifiable Information (PII) without retraining the model. Our approach consists of a four-step procedure: detecting memorized PII, applying PAE cards to mitigate memorization of private data, verifying resilience to targeted data extraction (TDE) attacks, and ensuring consistency in the post-edit LLMs. The versatility and efficiency of PAE, which allows for batch modifications, significantly enhance data privacy in LLMs. Experimental results demonstrate the effectiveness of PAE in mitigating private data leakage. We believe PAE will serve as a critical tool in the ongoing effort to protect data privacy in LLMs, encouraging the development of safer models for real-world applications.



## **28. Poisoned LangChain: Jailbreak LLMs by LangChain**

cs.CL

6 pages,2 figures,This paper is a submission to ACM TURC. It has been  accepted by the editor of the organizer

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18122v1) [paper-pdf](http://arxiv.org/pdf/2406.18122v1)

**Authors**: Ziqiu Wang, Jun Liu, Shengkai Zhang, Yang Yang

**Abstract**: With the development of natural language processing (NLP), large language models (LLMs) are becoming increasingly popular. LLMs are integrating more into everyday life, raising public concerns about their security vulnerabilities. Consequently, the security of large language models is becoming critically important. Currently, the techniques for attacking and defending against LLMs are continuously evolving. One significant method type of attack is the jailbreak attack, which designed to evade model safety mechanisms and induce the generation of inappropriate content. Existing jailbreak attacks primarily rely on crafting inducement prompts for direct jailbreaks, which are less effective against large models with robust filtering and high comprehension abilities. Given the increasing demand for real-time capabilities in large language models, real-time updates and iterations of new knowledge have become essential. Retrieval-Augmented Generation (RAG), an advanced technique to compensate for the model's lack of new knowledge, is gradually becoming mainstream. As RAG enables the model to utilize external knowledge bases, it provides a new avenue for jailbreak attacks.   In this paper, we conduct the first work to propose the concept of indirect jailbreak and achieve Retrieval-Augmented Generation via LangChain. Building on this, we further design a novel method of indirect jailbreak attack, termed Poisoned-LangChain (PLC), which leverages a poisoned external knowledge base to interact with large language models, thereby causing the large models to generate malicious non-compliant dialogues.We tested this method on six different large language models across three major categories of jailbreak issues. The experiments demonstrate that PLC successfully implemented indirect jailbreak attacks under three different scenarios, achieving success rates of 88.56%, 79.04%, and 82.69% respectively.



## **29. Safely Learning with Private Data: A Federated Learning Framework for Large Language Model**

cs.CR

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14898v2) [paper-pdf](http://arxiv.org/pdf/2406.14898v2)

**Authors**: JiaYing Zheng, HaiNan Zhang, LingXiang Wang, WangJie Qiu, HongWei Zheng, ZhiMing Zheng

**Abstract**: Private data, being larger and quality-higher than public data, can greatly improve large language models (LLM). However, due to privacy concerns, this data is often dispersed in multiple silos, making its secure utilization for LLM training a challenge. Federated learning (FL) is an ideal solution for training models with distributed private data, but traditional frameworks like FedAvg are unsuitable for LLM due to their high computational demands on clients. An alternative, split learning, offloads most training parameters to the server while training embedding and output layers locally, making it more suitable for LLM. Nonetheless, it faces significant challenges in security and efficiency. Firstly, the gradients of embeddings are prone to attacks, leading to potential reverse engineering of private data. Furthermore, the server's limitation of handle only one client's training request at a time hinders parallel training, severely impacting training efficiency. In this paper, we propose a Federated Learning framework for LLM, named FL-GLM, which prevents data leakage caused by both server-side and peer-client attacks while improving training efficiency. Specifically, we first place the input block and output block on local client to prevent embedding gradient attacks from server. Secondly, we employ key-encryption during client-server communication to prevent reverse engineering attacks from peer-clients. Lastly, we employ optimization methods like client-batching or server-hierarchical, adopting different acceleration methods based on the actual computational capabilities of the server. Experimental results on NLU and generation tasks demonstrate that FL-GLM achieves comparable metrics to centralized chatGLM model, validating the effectiveness of our federated learning framework.



## **30. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.



## **31. Inherent Challenges of Post-Hoc Membership Inference for Large Language Models**

cs.CL

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17975v1) [paper-pdf](http://arxiv.org/pdf/2406.17975v1)

**Authors**: Matthieu Meeus, Shubham Jain, Marek Rei, Yves-Alexandre de Montjoye

**Abstract**: Large Language Models (LLMs) are often trained on vast amounts of undisclosed data, motivating the development of post-hoc Membership Inference Attacks (MIAs) to gain insight into their training data composition. However, in this paper, we identify inherent challenges in post-hoc MIA evaluation due to potential distribution shifts between collected member and non-member datasets. Using a simple bag-of-words classifier, we demonstrate that datasets used in recent post-hoc MIAs suffer from significant distribution shifts, in some cases achieving near-perfect distinction between members and non-members. This implies that previously reported high MIA performance may be largely attributable to these shifts rather than model memorization. We confirm that randomized, controlled setups eliminate such shifts and thus enable the development and fair evaluation of new MIAs. However, we note that such randomized setups are rarely available for the latest LLMs, making post-hoc data collection still required to infer membership for real-world LLMs. As a potential solution, we propose a Regression Discontinuity Design (RDD) approach for post-hoc data collection, which substantially mitigates distribution shifts. Evaluating various MIA methods on this RDD setup yields performance barely above random guessing, in stark contrast to previously reported results. Overall, our findings highlight the challenges in accurately measuring LLM memorization and the need for careful experimental design in (post-hoc) membership inference tasks.



## **32. CoSafe: Evaluating Large Language Model Safety in Multi-Turn Dialogue Coreference**

cs.CL

Submitted to EMNLP 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17626v1) [paper-pdf](http://arxiv.org/pdf/2406.17626v1)

**Authors**: Erxin Yu, Jing Li, Ming Liao, Siqi Wang, Zuchen Gao, Fei Mi, Lanqing Hong

**Abstract**: As large language models (LLMs) constantly evolve, ensuring their safety remains a critical research problem. Previous red-teaming approaches for LLM safety have primarily focused on single prompt attacks or goal hijacking. To the best of our knowledge, we are the first to study LLM safety in multi-turn dialogue coreference. We created a dataset of 1,400 questions across 14 categories, each featuring multi-turn coreference safety attacks. We then conducted detailed evaluations on five widely used open-source LLMs. The results indicated that under multi-turn coreference safety attacks, the highest attack success rate was 56% with the LLaMA2-Chat-7b model, while the lowest was 13.9% with the Mistral-7B-Instruct model. These findings highlight the safety vulnerabilities in LLMs during dialogue coreference interactions.



## **33. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

Repo: https://github.com/wjfu99/MIA-LLMs

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2311.06062v3) [paper-pdf](http://arxiv.org/pdf/2311.06062v3)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **34. BEEAR: Embedding-based Adversarial Removal of Safety Backdoors in Instruction-tuned Language Models**

cs.CR

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.17092v1) [paper-pdf](http://arxiv.org/pdf/2406.17092v1)

**Authors**: Yi Zeng, Weiyu Sun, Tran Ngoc Huynh, Dawn Song, Bo Li, Ruoxi Jia

**Abstract**: Safety backdoor attacks in large language models (LLMs) enable the stealthy triggering of unsafe behaviors while evading detection during normal interactions. The high dimensionality of potential triggers in the token space and the diverse range of malicious behaviors make this a critical challenge. We present BEEAR, a mitigation approach leveraging the insight that backdoor triggers induce relatively uniform drifts in the model's embedding space. Our bi-level optimization method identifies universal embedding perturbations that elicit unwanted behaviors and adjusts the model parameters to reinforce safe behaviors against these perturbations. Experiments show BEEAR reduces the success rate of RLHF time backdoor attacks from >95% to <1% and from 47% to 0% for instruction-tuning time backdoors targeting malicious code generation, without compromising model utility. Requiring only defender-defined safe and unwanted behaviors, BEEAR represents a step towards practical defenses against safety backdoors in LLMs, providing a foundation for further advancements in AI safety and security.



## **35. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2402.17012v3) [paper-pdf](http://arxiv.org/pdf/2402.17012v3)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.



## **36. Versatile Backdoor Attack with Visible, Semantic, Sample-Specific, and Compatible Triggers**

cs.CV

23 pages, 21 figures, 18 tables

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2306.00816v4) [paper-pdf](http://arxiv.org/pdf/2306.00816v4)

**Authors**: Ruotong Wang, Hongrui Chen, Zihao Zhu, Li Liu, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) can be manipulated to exhibit specific behaviors when exposed to specific trigger patterns, without affecting their performance on benign samples, dubbed \textit{backdoor attack}. Currently, implementing backdoor attacks in physical scenarios still faces significant challenges. Physical attacks are labor-intensive and time-consuming, and the triggers are selected in a manual and heuristic way. Moreover, expanding digital attacks to physical scenarios faces many challenges due to their sensitivity to visual distortions and the absence of counterparts in the real world. To address these challenges, we define a novel trigger called the \textbf{V}isible, \textbf{S}emantic, \textbf{S}ample-Specific, and \textbf{C}ompatible (VSSC) trigger, to achieve effective, stealthy and robust simultaneously, which can also be effectively deployed in the physical scenario using corresponding objects. To implement the VSSC trigger, we propose an automated pipeline comprising three modules: a trigger selection module that systematically identifies suitable triggers leveraging large language models, a trigger insertion module that employs generative models to seamlessly integrate triggers into images, and a quality assessment module that ensures the natural and successful insertion of triggers through vision-language models. Extensive experimental results and analysis validate the effectiveness, stealthiness, and robustness of the VSSC trigger. It can not only maintain robustness under visual distortions but also demonstrates strong practicality in the physical scenario. We hope that the proposed VSSC trigger and implementation approach could inspire future studies on designing more practical triggers in backdoor attacks.



## **37. ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods**

cs.CL

**SubmitDate**: 2024-06-23    [abs](http://arxiv.org/abs/2406.15968v1) [paper-pdf](http://arxiv.org/pdf/2406.15968v1)

**Authors**: Roy Xie, Junlin Wang, Ruomin Huang, Minxing Zhang, Rong Ge, Jian Pei, Neil Zhenqiang Gong, Bhuwan Dhingra

**Abstract**: The rapid scaling of large language models (LLMs) has raised concerns about the transparency and fair use of the pretraining data used for training them. Detecting such content is challenging due to the scale of the data and limited exposure of each instance during training. We propose ReCaLL (Relative Conditional Log-Likelihood), a novel membership inference attack (MIA) to detect LLMs' pretraining data by leveraging their conditional language modeling capabilities. ReCaLL examines the relative change in conditional log-likelihoods when prefixing target data points with non-member context. Our empirical findings show that conditioning member data on non-member prefixes induces a larger decrease in log-likelihood compared to non-member data. We conduct comprehensive experiments and show that ReCaLL achieves state-of-the-art performance on the WikiMIA dataset, even with random and synthetic prefixes, and can be further improved using an ensemble approach. Moreover, we conduct an in-depth analysis of LLMs' behavior with different membership contexts, providing insights into how LLMs leverage membership information for effective inference at both the sequence and token level.



## **38. Large Language Models for Link Stealing Attacks Against Graph Neural Networks**

cs.LG

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2406.16963v1) [paper-pdf](http://arxiv.org/pdf/2406.16963v1)

**Authors**: Faqian Guan, Tianqing Zhu, Hui Sun, Wanlei Zhou, Philip S. Yu

**Abstract**: Graph data contains rich node features and unique edge information, which have been applied across various domains, such as citation networks or recommendation systems. Graph Neural Networks (GNNs) are specialized for handling such data and have shown impressive performance in many applications. However, GNNs may contain of sensitive information and susceptible to privacy attacks. For example, link stealing is a type of attack in which attackers infer whether two nodes are linked or not. Previous link stealing attacks primarily relied on posterior probabilities from the target GNN model, neglecting the significance of node features. Additionally, variations in node classes across different datasets lead to different dimensions of posterior probabilities. The handling of these varying data dimensions posed a challenge in using a single model to effectively conduct link stealing attacks on different datasets. To address these challenges, we introduce Large Language Models (LLMs) to perform link stealing attacks on GNNs. LLMs can effectively integrate textual features and exhibit strong generalizability, enabling attacks to handle diverse data dimensions across various datasets. We design two distinct LLM prompts to effectively combine textual features and posterior probabilities of graph nodes. Through these designed prompts, we fine-tune the LLM to adapt to the link stealing attack task. Furthermore, we fine-tune the LLM using multiple datasets and enable the LLM to learn features from different datasets simultaneously. Experimental results show that our approach significantly enhances the performance of existing link stealing attack tasks in both white-box and black-box scenarios. Our method can execute link stealing attacks across different datasets using only a single model, making link stealing attacks more applicable to real-world scenarios.



## **39. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2405.15589v2) [paper-pdf](http://arxiv.org/pdf/2405.15589v2)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **40. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2407.00075v1) [paper-pdf](http://arxiv.org/pdf/2407.00075v1)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert language models from following the rules. We model rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. We prove that although transformers can faithfully abide by such rules, maliciously crafted prompts can nevertheless mislead even theoretically constructed models. Empirically, we find that attacks on our theoretical models mirror popular attacks on large language models. Our work suggests that studying smaller theoretical models can help understand the behavior of large language models in rule-based settings like logical reasoning and jailbreak attacks.



## **41. From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking**

cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14859v1) [paper-pdf](http://arxiv.org/pdf/2406.14859v1)

**Authors**: Siyuan Wang, Zhuohan Long, Zhihao Fan, Zhongyu Wei

**Abstract**: The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.



## **42. FedSecurity: Benchmarking Attacks and Defenses in Federated Learning and Federated LLMs**

cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2306.04959v5) [paper-pdf](http://arxiv.org/pdf/2306.04959v5)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark that serves as a supplementary component of the FedML library for simulating adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity eliminates the need for implementing the fundamental FL procedures, e.g., FL training and data loading, from scratch, thus enables users to focus on developing their own attack and defense strategies. It contains two key components, including FedAttacker that conducts a variety of attacks during FL training, and FedDefender that implements defensive mechanisms to counteract these attacks. FedSecurity has the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs) to showcase its potential on a wide range of complex applications.



## **43. Unmasking Database Vulnerabilities: Zero-Knowledge Schema Inference Attacks in Text-to-SQL Systems**

cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14545v1) [paper-pdf](http://arxiv.org/pdf/2406.14545v1)

**Authors**: Đorđe Klisura, Anthony Rios

**Abstract**: Relational databases are integral to modern information systems, serving as the foundation for storing, querying, and managing data efficiently and effectively. Advancements in large language modeling have led to the emergence of text-to-SQL technologies, significantly enhancing the querying and extracting of information from these databases and raising concerns about privacy and security. Our research extracts the database schema elements underlying a text-to-SQL model. Knowledge of the schema can make attacks such as SQL injection easier. By asking specially crafted questions, we have developed a zero-knowledge framework designed to probe various database schema elements without knowledge of the database itself. The text-to-SQL models then process these questions to produce an output that we use to uncover the structure of the database schema. We apply it to specialized text-to-SQL models fine-tuned on text-SQL pairs and generative language models used for SQL generation. Overall, we can reconstruct the table names with an F1 of nearly .75 for fine-tuned models and .96 for generative.



## **44. Jailbreaking as a Reward Misspecification Problem**

cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14393v1) [paper-pdf](http://arxiv.org/pdf/2406.14393v1)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.



## **45. Safety of Multimodal Large Language Models on Images and Texts**

cs.CV

Accepted at IJCAI2024

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.00357v3) [paper-pdf](http://arxiv.org/pdf/2402.00357v3)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Attracted by the impressive power of Multimodal Large Language Models (MLLMs), the public is increasingly utilizing them to improve the efficiency of daily work. Nonetheless, the vulnerabilities of MLLMs to unsafe instructions bring huge safety risks when these models are deployed in real-world scenarios. In this paper, we systematically survey current efforts on the evaluation, attack, and defense of MLLMs' safety on images and text. We begin with introducing the overview of MLLMs on images and text and understanding of safety, which helps researchers know the detailed scope of our survey. Then, we review the evaluation datasets and metrics for measuring the safety of MLLMs. Next, we comprehensively present attack and defense techniques related to MLLMs' safety. Finally, we analyze several unsolved issues and discuss promising research directions. The latest papers are continually collected at https://github.com/isXinLiu/MLLM-Safety-Collection.



## **46. Are you still on track!? Catching LLM Task Drift with Activations**

cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.00799v3) [paper-pdf](http://arxiv.org/pdf/2406.00799v3)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models (LLMs) are routinely used in retrieval-augmented applications to orchestrate tasks and process inputs from users and other sources. These inputs, even in a single LLM interaction, can come from a variety of sources, of varying trustworthiness and provenance. This opens the door to prompt injection attacks, where the LLM receives and acts upon instructions from supposedly data-only sources, thus deviating from the user's original instructions. We define this as task drift, and we propose to catch it by scanning and analyzing the LLM's activations. We compare the LLM's activations before and after processing the external input in order to detect whether this input caused instruction drift. We develop two probing methods and find that simply using a linear classifier can detect drift with near perfect ROC AUC on an out-of-distribution test set. We show that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Our setup does not require any modification of the LLM (e.g., fine-tuning) or any text generation, thus maximizing deployability and cost efficiency and avoiding reliance on unreliable model output. To foster future research on activation-based task inspection, decoding, and interpretability, we will release our large-scale TaskTracker toolkit, comprising a dataset of over 500K instances, representations from 4 SoTA language models, and inspection tools.



## **47. FewFedPIT: Towards Privacy-preserving and Few-shot Federated Instruction Tuning**

cs.CR

Work in progress

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2403.06131v2) [paper-pdf](http://arxiv.org/pdf/2403.06131v2)

**Authors**: Zhuo Zhang, Jingyuan Zhang, Jintao Huang, Lizhen Qu, Hongzhi Zhang, Qifan Wang, Xun Zhou, Zenglin Xu

**Abstract**: Instruction tuning has been identified as a crucial technique for optimizing the performance of large language models (LLMs) in generating human-aligned responses. Nonetheless, gathering diversified and superior-quality instruction data for such tuning presents notable obstacles, especially in domains with rigid privacy provisions. Federated instruction tuning (FedIT) has emerged as a promising solution, by consolidating collaborative training across multiple data owners, thereby resulting in a privacy-preserving learning model. However, FedIT encounters limitations such as scarcity of instructional data and risk of exposure to training data extraction attacks. In this paper, we propose a novel federated algorithm, FewFedPIT, designed to simultaneously enhance privacy protection and model performance of federated few-shot learning. FewFedPITcomprises three vital components on the client side: (1) synthetic data generation, which utilizes LLMs' in-context learning capacity to generate synthetic data autonomously, thus expanding the local database; (2) parameter isolation training, which individually updates the public parameters in the synthetic data and the private parameters in the local data, consequently mitigating the noise impact of the synthetic data; (3) local aggregation sharing, which mixes public and private parameters before uploading, effectively preventing data extraction attacks. Extensive experiments on three open-source datasets demonstrate the effectiveness of FewFedPITin, enhancing privacy preservation and improving federated few-shot performance.



## **48. Protecting Privacy Through Approximating Optimal Parameters for Sequence Unlearning in Language Models**

cs.CL

Accepted to ACL2024 findings

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14091v1) [paper-pdf](http://arxiv.org/pdf/2406.14091v1)

**Authors**: Dohyun Lee, Daniel Rim, Minseok Choi, Jaegul Choo

**Abstract**: Although language models (LMs) demonstrate exceptional capabilities on various tasks, they are potentially vulnerable to extraction attacks, which represent a significant privacy risk. To mitigate the privacy concerns of LMs, machine unlearning has emerged as an important research area, which is utilized to induce the LM to selectively forget about some of its training data. While completely retraining the model will guarantee successful unlearning and privacy assurance, it is impractical for LMs, as it would be time-consuming and resource-intensive. Prior works efficiently unlearn the target token sequences, but upon subsequent iterations, the LM displays significant degradation in performance. In this work, we propose Privacy Protection via Optimal Parameters (POP), a novel unlearning method that effectively forgets the target token sequences from the pretrained LM by applying optimal gradient updates to the parameters. Inspired by the gradient derivation of complete retraining, we approximate the optimal training objective that successfully unlearns the target sequence while retaining the knowledge from the rest of the training data. Experimental results demonstrate that POP exhibits remarkable retention performance post-unlearning across 9 classification and 4 dialogue benchmarks, outperforming the state-of-the-art by a large margin. Furthermore, we introduce Remnant Memorization Accuracy that quantifies privacy risks based on token likelihood and validate its effectiveness through both qualitative and quantitative analyses.



## **49. Prompt Injection Attacks in Defended Systems**

cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14048v1) [paper-pdf](http://arxiv.org/pdf/2406.14048v1)

**Authors**: Daniil Khomsky, Narek Maloyan, Bulat Nutfullin

**Abstract**: Large language models play a crucial role in modern natural language processing technologies. However, their extensive use also introduces potential security risks, such as the possibility of black-box attacks. These attacks can embed hidden malicious features into the model, leading to adverse consequences during its deployment.   This paper investigates methods for black-box attacks on large language models with a three-tiered defense mechanism. It analyzes the challenges and significance of these attacks, highlighting their potential implications for language processing system security. Existing attack and defense methods are examined, evaluating their effectiveness and applicability across various scenarios.   Special attention is given to the detection algorithm for black-box attacks, identifying hazardous vulnerabilities in language models and retrieving sensitive information. This research presents a methodology for vulnerability detection and the development of defensive strategies against black-box attacks on large language models.



## **50. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

cs.CL

Code and datasets are available at  https://github.com/wen112358/ImplicitBiasPsychometricEvaluation

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14023v1) [paper-pdf](http://arxiv.org/pdf/2406.14023v1)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As Large Language Models (LLMs) become an important way of information seeking, there have been increasing concerns about the unethical content LLMs may generate. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain groups by attacking them with carefully crafted instructions to elicit biased responses. Our attack methodology is inspired by psychometric principles in cognitive and social psychology. We propose three attack approaches, i.e., Disguise, Deception, and Teaching, based on which we built evaluation datasets for four common bias types. Each prompt attack has bilingual versions. Extensive evaluation of representative LLMs shows that 1) all three attack methods work effectively, especially the Deception attacks; 2) GLM-3 performs the best in defending our attacks, compared to GPT-3.5 and GPT-4; 3) LLMs could output content of other bias types when being taught with one type of bias. Our methodology provides a rigorous and effective way of evaluating LLMs' implicit bias and will benefit the assessments of LLMs' potential ethical risks.



