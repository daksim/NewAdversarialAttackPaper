# Latest Large Language Model Attack Papers
**update at 2024-11-11 09:19:36**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Fine-tuning Large Language Models for DGA and DNS Exfiltration Detection**

cs.CR

Accepted in Proceedings of the Workshop at AI for Cyber Threat  Intelligence (WAITI), 2024

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2410.21723v2) [paper-pdf](http://arxiv.org/pdf/2410.21723v2)

**Authors**: Md Abu Sayed, Asif Rahman, Christopher Kiekintveld, Sebastian Garcia

**Abstract**: Domain Generation Algorithms (DGAs) are malicious techniques used by malware to dynamically generate seemingly random domain names for communication with Command & Control (C&C) servers. Due to the fast and simple generation of DGA domains, detection methods must be highly efficient and precise to be effective. Large Language Models (LLMs) have demonstrated their proficiency in real-time detection tasks, making them ideal candidates for detecting DGAs. Our work validates the effectiveness of fine-tuned LLMs for detecting DGAs and DNS exfiltration attacks. We developed LLM models and conducted comprehensive evaluation using a diverse dataset comprising 59 distinct real-world DGA malware families and normal domain data. Our LLM model significantly outperformed traditional natural language processing techniques, especially in detecting unknown DGAs. We also evaluated its performance on DNS exfiltration datasets, demonstrating its effectiveness in enhancing cybersecurity measures. To the best of our knowledge, this is the first work that empirically applies LLMs for DGA and DNS exfiltration detection.



## **2. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.



## **3. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

cs.CR

Accepted by NeurIPS 2024. Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2403.00867v3) [paper-pdf](http://arxiv.org/pdf/2403.00867v3)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.



## **4. Unfair Alignment: Examining Safety Alignment Across Vision Encoder Layers in Vision-Language Models**

cs.CL

Preprint, Under Review

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.04291v1) [paper-pdf](http://arxiv.org/pdf/2411.04291v1)

**Authors**: Saketh Bachu, Erfan Shayegani, Trishna Chakraborty, Rohit Lal, Arindam Dutta, Chengyu Song, Yue Dong, Nael Abu-Ghazaleh, Amit K. Roy-Chowdhury

**Abstract**: Vision-language models (VLMs) have improved significantly in multi-modal tasks, but their more complex architecture makes their safety alignment more challenging than the alignment of large language models (LLMs). In this paper, we reveal an unfair distribution of safety across the layers of VLM's vision encoder, with earlier and middle layers being disproportionately vulnerable to malicious inputs compared to the more robust final layers. This 'cross-layer' vulnerability stems from the model's inability to generalize its safety training from the default architectural settings used during training to unseen or out-of-distribution scenarios, leaving certain layers exposed. We conduct a comprehensive analysis by projecting activations from various intermediate layers and demonstrate that these layers are more likely to generate harmful outputs when exposed to malicious inputs. Our experiments with LLaVA-1.5 and Llama 3.2 show discrepancies in attack success rates and toxicity scores across layers, indicating that current safety alignment strategies focused on a single default layer are insufficient.



## **5. Diversity Helps Jailbreak Large Language Models**

cs.CL

arXiv admin note: text overlap with arXiv:2312.02119

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.04223v1) [paper-pdf](http://arxiv.org/pdf/2411.04223v1)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62% higher success rate in compromising nine leading chatbots, including GPT-4, Gemini, and Llama, while using only 13% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.



## **6. MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue**

cs.AI

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2411.03814v1) [paper-pdf](http://arxiv.org/pdf/2411.03814v1)

**Authors**: Fengxiang Wang, Ranjie Duan, Peng Xiao, Xiaojun Jia, YueFeng Chen, Chongwen Wang, Jialing Tao, Hang Su, Jun Zhu, Hui Xue

**Abstract**: Large Language Models (LLMs) demonstrate outstanding performance in their reservoir of knowledge and understanding capabilities, but they have also been shown to be prone to illegal or unethical reactions when subjected to jailbreak attacks. To ensure their responsible deployment in critical applications, it is crucial to understand the safety capabilities and vulnerabilities of LLMs. Previous works mainly focus on jailbreak in single-round dialogue, overlooking the potential jailbreak risks in multi-round dialogues, which are a vital way humans interact with and extract information from LLMs. Some studies have increasingly concentrated on the risks associated with jailbreak in multi-round dialogues. These efforts typically involve the use of manually crafted templates or prompt engineering techniques. However, due to the inherent complexity of multi-round dialogues, their jailbreak performance is limited. To solve this problem, we propose a novel multi-round dialogue jailbreaking agent, emphasizing the importance of stealthiness in identifying and mitigating potential threats to human values posed by LLMs. We propose a risk decomposition strategy that distributes risks across multiple rounds of queries and utilizes psychological strategies to enhance attack strength. Extensive experiments show that our proposed method surpasses other attack methods and achieves state-of-the-art attack success rate. We will make the corresponding code and dataset available for future research. The code will be released soon.



## **7. Whispers in the Machine: Confidentiality in LLM-integrated Systems**

cs.CR

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2402.06922v3) [paper-pdf](http://arxiv.org/pdf/2402.06922v3)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: Large Language Models (LLMs) are increasingly augmented with external tools and commercial services into LLM-integrated systems. While these interfaces can significantly enhance the capabilities of the models, they also introduce a new attack surface. Manipulated integrations, for example, can exploit the model and compromise sensitive data accessed through other interfaces. While previous work primarily focused on attacks targeting a model's alignment or the leakage of training data, the security of data that is only available during inference has escaped scrutiny so far. In this work, we demonstrate the vulnerabilities associated with external components and introduce a systematic approach to evaluate confidentiality risks in LLM-integrated systems. We identify two specific attack scenarios unique to these systems and formalize these into a tool-robustness framework designed to measure a model's ability to protect sensitive information. Our findings show that all examined models are highly vulnerable to confidentiality attacks, with the risk increasing significantly when models are used together with external tools.



## **8. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2409.20002v2) [paper-pdf](http://arxiv.org/pdf/2409.20002v2)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **9. Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs**

cs.CR

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2406.09324v3) [paper-pdf](http://arxiv.org/pdf/2406.09324v3)

**Authors**: Zhao Xu, Fan Liu, Hao Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. Recently, a growing body of research has categorized jailbreak attacks into token-level and prompt-level attacks. However, previous work primarily overlooks the diverse key factors of jailbreak attacks, with most studies concentrating on LLM vulnerabilities and lacking exploration of defense-enhanced LLMs. To address these issues, we introduced $\textbf{JailTrickBench}$ to evaluate the impact of various attack settings on LLM performance and provide a baseline for jailbreak attacks, encouraging the adoption of a standardized evaluation framework. Specifically, we evaluate the eight key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. We further conduct seven representative jailbreak attacks on six defense methods across two widely used datasets, encompassing approximately 354 experiments with about 55,000 GPU hours on A800-80G. Our experimental results highlight the need for standardized benchmarking to evaluate these attacks on defense-enhanced LLMs. Our code is available at https://github.com/usail-hkust/JailTrickBench.



## **10. Jailbreaking Large Language Models with Symbolic Mathematics**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.11445v2) [paper-pdf](http://arxiv.org/pdf/2409.11445v2)

**Authors**: Emet Bethany, Mazal Bethany, Juan Arturo Nolazco Flores, Sumit Kumar Jha, Peyman Najafirad

**Abstract**: Recent advancements in AI safety have led to increased efforts in training and red-teaming large language models (LLMs) to mitigate unsafe content generation. However, these safety mechanisms may not be comprehensive, leaving potential vulnerabilities unexplored. This paper introduces MathPrompt, a novel jailbreaking technique that exploits LLMs' advanced capabilities in symbolic mathematics to bypass their safety mechanisms. By encoding harmful natural language prompts into mathematical problems, we demonstrate a critical vulnerability in current AI safety measures. Our experiments across 13 state-of-the-art LLMs reveal an average attack success rate of 73.6\%, highlighting the inability of existing safety training mechanisms to generalize to mathematically encoded inputs. Analysis of embedding vectors shows a substantial semantic shift between original and encoded prompts, helping explain the attack's success. This work emphasizes the importance of a holistic approach to AI safety, calling for expanded red-teaming efforts to develop robust safeguards across all potential input types and their associated risks.



## **11. Membership Inference Attacks against Large Vision-Language Models**

cs.CV

NeurIPS 2024

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02902v1) [paper-pdf](http://arxiv.org/pdf/2411.02902v1)

**Authors**: Zhan Li, Yongtao Wu, Yihang Chen, Francesco Tonin, Elias Abad Rocamora, Volkan Cevher

**Abstract**: Large vision-language models (VLLMs) exhibit promising capabilities for processing multi-modal tasks across various application scenarios. However, their emergence also raises significant data security concerns, given the potential inclusion of sensitive information, such as private photos and medical records, in their training datasets. Detecting inappropriately used data in VLLMs remains a critical and unresolved issue, mainly due to the lack of standardized datasets and suitable methodologies. In this study, we introduce the first membership inference attack (MIA) benchmark tailored for various VLLMs to facilitate training data detection. Then, we propose a novel MIA pipeline specifically designed for token-level image detection. Lastly, we present a new metric called MaxR\'enyi-K%, which is based on the confidence of the model output and applies to both text and image data. We believe that our work can deepen the understanding and methodology of MIAs in the context of VLLMs. Our code and datasets are available at https://github.com/LIONS-EPFL/VL-MIA.



## **12. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.



## **13. Extracting Unlearned Information from LLMs with Activation Steering**

cs.CL

Accepted at NeurIPS 2024 Workshop Safe Generative AI

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02631v1) [paper-pdf](http://arxiv.org/pdf/2411.02631v1)

**Authors**: Atakan Seyitoğlu, Aleksei Kuvshinov, Leo Schwinn, Stephan Günnemann

**Abstract**: An unintended consequence of the vast pretraining of Large Language Models (LLMs) is the verbatim memorization of fragments of their training data, which may contain sensitive or copyrighted information. In recent years, unlearning has emerged as a solution to effectively remove sensitive knowledge from models after training. Yet, recent work has shown that supposedly deleted information can still be extracted by malicious actors through various attacks. Still, current attacks retrieve sets of possible candidate generations and are unable to pinpoint the output that contains the actual target information. We propose activation steering as a method for exact information retrieval from unlearned LLMs. We introduce a novel approach to generating steering vectors, named Anonymized Activation Steering. Additionally, we develop a simple word frequency method to pinpoint the correct answer among a set of candidates when retrieving unlearned information. Our evaluation across multiple unlearning techniques and datasets demonstrates that activation steering successfully recovers general knowledge (e.g., widely known fictional characters) while revealing limitations in retrieving specific information (e.g., details about non-public individuals). Overall, our results demonstrate that exact information retrieval from unlearned models is possible, highlighting a severe vulnerability of current unlearning techniques.



## **14. Attacking Vision-Language Computer Agents via Pop-ups**

cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.



## **15. Defining and Evaluating Physical Safety for Large Language Models**

cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02317v1) [paper-pdf](http://arxiv.org/pdf/2411.02317v1)

**Authors**: Yung-Chen Tang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly used to control robotic systems such as drones, but their risks of causing physical threats and harm in real-world applications remain unexplored. Our study addresses the critical gap in evaluating LLM physical safety by developing a comprehensive benchmark for drone control. We classify the physical safety risks of drones into four categories: (1) human-targeted threats, (2) object-targeted threats, (3) infrastructure attacks, and (4) regulatory violations. Our evaluation of mainstream LLMs reveals an undesirable trade-off between utility and safety, with models that excel in code generation often performing poorly in crucial safety aspects. Furthermore, while incorporating advanced prompt engineering techniques such as In-Context Learning and Chain-of-Thought can improve safety, these methods still struggle to identify unintentional attacks. In addition, larger models demonstrate better safety capabilities, particularly in refusing dangerous commands. Our findings and benchmark can facilitate the design and evaluation of physical safety for LLMs. The project page is available at huggingface.co/spaces/TrustSafeAI/LLM-physical-safety.



## **16. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **17. Exploiting LLM Quantization**

cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.



## **18. Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors**

cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01705v1) [paper-pdf](http://arxiv.org/pdf/2411.01705v1)

**Authors**: Yuefeng Peng, Junda Wang, Hong Yu, Amir Houmansadr

**Abstract**: Despite significant advancements, large language models (LLMs) still struggle with providing accurate answers when lacking domain-specific or up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge bases, but it also introduces new attack surfaces. In this paper, we investigate data extraction attacks targeting the knowledge databases of RAG systems. We demonstrate that previous attacks on RAG largely depend on the instruction-following capabilities of LLMs, and that simple fine-tuning can reduce the success rate of such attacks to nearly zero. This makes these attacks impractical since fine-tuning is a common practice when deploying LLMs in specific domains. To further reveal the vulnerability, we propose to backdoor RAG, where a small portion of poisoned data is injected during the fine-tuning phase to create a backdoor within the LLM. When this compromised LLM is integrated into a RAG system, attackers can exploit specific triggers in prompts to manipulate the LLM to leak documents from the retrieval database. By carefully designing the poisoned data, we achieve both verbatim and paraphrased document extraction. We show that with only 3\% poisoned data, our method achieves an average success rate of 79.7\% in verbatim extraction on Llama2-7B, with a ROUGE-L score of 64.21, and a 68.6\% average success rate in paraphrased extraction, with an average ROUGE score of 52.6 across four datasets. These results underscore the privacy risks associated with the supply chain when deploying RAG systems.



## **19. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.



## **20. CARES: A Comprehensive Benchmark of Trustworthiness in Medical Vision Language Models**

cs.LG

NeurIPS 2024 Datasets and Benchmarks Track

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2406.06007v3) [paper-pdf](http://arxiv.org/pdf/2406.06007v3)

**Authors**: Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, Wenhao Zheng, Zhaoyang Wang, Xiao Wang, Xuchao Zhang, Chetan Bansal, Marc Niethammer, Junzhou Huang, Hongtu Zhu, Yun Li, Jimeng Sun, Zongyuan Ge, Gang Li, James Zou, Huaxiu Yao

**Abstract**: Artificial intelligence has significantly impacted medical applications, particularly with the advent of Medical Large Vision Language Models (Med-LVLMs), sparking optimism for the future of automated and personalized healthcare. However, the trustworthiness of Med-LVLMs remains unverified, posing significant risks for future model deployment. In this paper, we introduce CARES and aim to comprehensively evaluate the Trustworthiness of Med-LVLMs across the medical domain. We assess the trustworthiness of Med-LVLMs across five dimensions, including trustfulness, fairness, safety, privacy, and robustness. CARES comprises about 41K question-answer pairs in both closed and open-ended formats, covering 16 medical image modalities and 27 anatomical regions. Our analysis reveals that the models consistently exhibit concerns regarding trustworthiness, often displaying factual inaccuracies and failing to maintain fairness across different demographic groups. Furthermore, they are vulnerable to attacks and demonstrate a lack of privacy awareness. We publicly release our benchmark and code in https://cares-ai.github.io/.



## **21. Are you still on track!? Catching LLM Task Drift with Activations**

cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2406.00799v5) [paper-pdf](http://arxiv.org/pdf/2406.00799v5)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: Large Language Models are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how user's tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and inspection tools.



## **22. SQL Injection Jailbreak: a structural disaster of large language models**

cs.CR

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01565v1) [paper-pdf](http://arxiv.org/pdf/2411.01565v1)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: In recent years, the rapid development of large language models (LLMs) has brought new vitality to the various domains and generated substantial social and economic benefits. However, the swift advancement of LLMs has introduced new security vulnerabilities. Jailbreak, a form of attack that induces LLMs to output harmful content through carefully crafted prompts, poses a challenge to the safe and trustworthy development of LLMs. Previous jailbreak attack methods primarily exploited the internal capabilities of the model. Among them, one category leverages the model's implicit capabilities for jailbreak attacks, where the attacker is unaware of the exact reasons for the attack's success. The other category utilizes the model's explicit capabilities for jailbreak attacks, where the attacker understands the reasons for the attack's success. For example, these attacks exploit the model's abilities in coding, contextual learning, or understanding ASCII characters. However, these earlier jailbreak attacks have certain limitations, as they only exploit the inherent capabilities of the model. In this paper, we propose a novel jailbreak method, SQL Injection Jailbreak (SIJ), which utilizes the construction of input prompts by LLMs to inject jailbreak information into user prompts, enabling successful jailbreak of the LLMs. Our SIJ method achieves nearly 100\% attack success rates on five well-known open-source LLMs in the context of AdvBench, while incurring lower time costs compared to previous methods. More importantly, SIJ reveals a new vulnerability in LLMs that urgently needs to be addressed. To this end, we propose a defense method called Self-Reminder-Key and demonstrate its effectiveness through experiments. Our code is available at \href{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}{https://github.com/weiyezhimeng/SQL-Injection-Jailbreak}.



## **23. Boosting Jailbreak Transferability for Large Language Models**

cs.AI

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2410.15645v2) [paper-pdf](http://arxiv.org/pdf/2410.15645v2)

**Authors**: Hanqing Liu, Lifeng Zhou, Huanqian Yan

**Abstract**: Large language models have drawn significant attention to the challenge of safe alignment, especially regarding jailbreak attacks that circumvent security measures to produce harmful content. To address the limitations of existing methods like GCG, which perform well in single-model attacks but lack transferability, we propose several enhancements, including a scenario induction template, optimized suffix selection, and the integration of re-suffix attack mechanism to reduce inconsistent outputs. Our approach has shown superior performance in extensive experiments across various benchmarks, achieving nearly 100% success rates in both attack execution and transferability. Notably, our method has won the first place in the AISG-hosted Global Challenge for Safe and Secure LLMs. The code is released at https://github.com/HqingLiu/SI-GCG.



## **24. Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs**

cs.CL

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2410.24049v2) [paper-pdf](http://arxiv.org/pdf/2410.24049v2)

**Authors**: Muhammed Saeed, Elgizouli Mohamed, Mukhtar Mohamed, Shaina Raza, Shady Shehata, Muhammad Abdul-Mageed

**Abstract**: Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.



## **25. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

cs.CR

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2411.03343v1) [paper-pdf](http://arxiv.org/pdf/2411.03343v1)

**Authors**: Nathalie Maria Kirch, Severin Field, Stephen Casper

**Abstract**: While `jailbreaks' have been central to research on the safety and reliability of LLMs (large language models), the underlying mechanisms behind these attacks are not well understood. Some prior works have used linear methods to analyze jailbreak prompts or model refusal. Here, however, we compare linear and nonlinear methods to study the features in prompts that contribute to successful jailbreaks. We do this by probing for jailbreak success based only on the portions of the latent representations corresponding to prompt tokens. First, we introduce a dataset of 10,800 jailbreak attempts from 35 attack methods. We then show that different jailbreaking methods work via different nonlinear features in prompts. Specifically, we find that while probes can distinguish between successful and unsuccessful jailbreaking prompts with a high degree of accuracy, they often transfer poorly to held-out attack methods. We also show that nonlinear probes can be used to mechanistically jailbreak the LLM by guiding the design of adversarial latent perturbations. These mechanistic jailbreaks are able to jailbreak Gemma-7B-IT more reliably than 34 of the 35 techniques that it was trained on. Ultimately, our results suggest that jailbreaks cannot be thoroughly understood in terms of universal or linear prompt features alone.



## **26. Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding**

cs.AI

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2406.15481v2) [paper-pdf](http://arxiv.org/pdf/2406.15481v2)

**Authors**: Haneul Yoo, Yongjin Yang, Hwaran Lee

**Abstract**: As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize code-switching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand code-switching texts. Additionally, we validate the extensibility of the CSRT by generating code-switching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.



## **27. HuRef: HUman-REadable Fingerprint for Large Language Models**

cs.CL

NeurIPS 2024

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2312.04828v4) [paper-pdf](http://arxiv.org/pdf/2312.04828v4)

**Authors**: Boyi Zeng, Lizheng Wang, Yuncong Hu, Yi Xu, Chenghu Zhou, Xinbing Wang, Yu Yu, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without interfering with training or exposing model parameters to the public. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, with negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning, and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. Due to the potential risk of information leakage, we cannot publish invariant terms directly. Instead, we map them to a Gaussian vector using an encoder, then convert it into a natural image using StyleGAN2, and finally publish the image. In our black-box setting, all fingerprinting steps are internally conducted by the LLMs owners. To ensure the published fingerprints are honestly generated, we introduced Zero-Knowledge Proof (ZKP). Experimental results across various LLMs demonstrate the effectiveness of our method. The code is available at https://github.com/LUMIA-Group/HuRef.



## **28. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01084v1) [paper-pdf](http://arxiv.org/pdf/2411.01084v1)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or \textit{red-teamers}, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary \textit{string compositions}, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



## **29. Emoji Attack: A Method for Misleading Judge LLMs in Safety Risk Detection**

cs.CL

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01077v1) [paper-pdf](http://arxiv.org/pdf/2411.01077v1)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking attacks show how Large Language Models (LLMs) can be tricked into generating harmful outputs using malicious prompts. To prevent these attacks, other LLMs are often used as judges to evaluate the harmfulness of the generated content. However, relying on LLMs as judges can introduce biases into the detection process, which in turn compromises the effectiveness of the evaluation. In this paper, we show that Judge LLMs, like other LLMs, are also affected by token segmentation bias. This bias occurs when tokens are split into smaller sub-tokens, altering their embeddings. This makes it harder for the model to detect harmful content. Specifically, this bias can cause sub-tokens to differ significantly from the original token in the embedding space, leading to incorrect "safe" predictions for harmful content. To exploit this bias in Judge LLMs, we introduce the Emoji Attack -- a method that places emojis within tokens to increase the embedding differences between sub-tokens and their originals. These emojis create new tokens that further distort the token embeddings, exacerbating the bias. To counter the Emoji Attack, we design prompts that help LLMs filter out unusual characters. However, this defense can still be bypassed by using a mix of emojis and other characters. The Emoji Attack can also be combined with existing jailbreaking prompts using few-shot learning, which enables LLMs to generate harmful responses with emojis. These responses are often mistakenly labeled as "safe" by Judge LLMs, allowing the attack to slip through. Our experiments with six state-of-the-art Judge LLMs show that the Emoji Attack allows 25\% of harmful responses to bypass detection by Llama Guard and Llama Guard 2, and up to 75\% by ShieldLM. These results highlight the need for stronger Judge LLMs to address this vulnerability.



## **30. FedDTPT: Federated Discrete and Transferable Prompt Tuning for Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00985v1) [paper-pdf](http://arxiv.org/pdf/2411.00985v1)

**Authors**: Jiaqi Wu, Simin Chen, Yuzhe Yang, Yijiang Li, Shiyue Hou, Rui Jing, Zehua Wang, Wei Chen, Zijian Tian

**Abstract**: In recent years, large language models (LLMs) have significantly advanced the field of natural language processing (NLP). By fine-tuning LLMs with data from specific scenarios, these foundation models can better adapt to various downstream tasks. However, the fine-tuning process poses privacy leakage risks, particularly in centralized data processing scenarios. To address user privacy concerns, federated learning (FL) has been introduced to mitigate the risks associated with centralized data collection from multiple sources. Nevertheless, the privacy of LLMs themselves is equally critical, as potential malicious attacks challenge their security, an issue that has received limited attention in current research. Consequently, establishing a trusted multi-party model fine-tuning environment is essential. Additionally, the local deployment of large LLMs incurs significant storage costs and high computational demands. To address these challenges, we propose for the first time a federated discrete and transferable prompt tuning, namely FedDTPT, for black-box large language models. In the client optimization phase, we adopt a token-level discrete prompt optimization method that leverages a feedback loop based on prediction accuracy to drive gradient-free prompt optimization through the MLM API. For server optimization, we employ an attention mechanism based on semantic similarity to filter all local prompt tokens, along with an embedding distance elbow detection and DBSCAN clustering strategy to enhance the filtering process. Experimental results demonstrate that, compared to state-of-the-art methods, our approach achieves higher accuracy, reduced communication overhead, and robustness to non-iid data in a black-box setting. Moreover, the optimized prompts are transferable.



## **31. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **32. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.



## **33. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

9 pages

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00459v1) [paper-pdf](http://arxiv.org/pdf/2411.00459v1)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **34. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00348v1) [paper-pdf](http://arxiv.org/pdf/2411.00348v1)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.



## **35. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **36. Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models**

cs.CL

Our code is available at https://github.com/parameterlab/mia-scaling

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00154v1) [paper-pdf](http://arxiv.org/pdf/2411.00154v1)

**Authors**: Haritz Puerto, Martin Gubri, Sangdoo Yun, Seong Joon Oh

**Abstract**: Membership inference attacks (MIA) attempt to verify the membership of a given data sample in the training set for a model. MIA has become relevant in recent years, following the rapid development of large language models (LLM). Many are concerned about the usage of copyrighted materials for training them and call for methods for detecting such usage. However, recent research has largely concluded that current MIA methods do not work on LLMs. Even when they seem to work, it is usually because of the ill-designed experimental setup where other shortcut features enable "cheating." In this work, we argue that MIA still works on LLMs, but only when multiple documents are presented for testing. We construct new benchmarks that measure the MIA performances at a continuous scale of data samples, from sentences (n-grams) to a collection of documents (multiple chunks of tokens). To validate the efficacy of current MIA approaches at greater scales, we adapt a recent work on Dataset Inference (DI) for the task of binary membership detection that aggregates paragraph-level MIA features to enable MIA at document and collection of documents level. This baseline achieves the first successful MIA on pre-trained and fine-tuned LLMs.



## **37. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

cs.LG

Accepted for presentation at NeurIPS 2024. Code:  https://github.com/RICommunity/TAP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2312.02119v3) [paper-pdf](http://arxiv.org/pdf/2312.02119v3)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an attacker LLM to iteratively refine candidate (attack) prompts until one of the refined prompts jailbreaks the target. In addition, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks, reducing the number of queries sent to the target LLM. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4-Turbo and GPT4o) for more than 80% of the prompts. This significantly improves upon the previous state-of-the-art black-box methods for generating jailbreaks while using a smaller number of queries than them. Furthermore, TAP is also capable of jailbreaking LLMs protected by state-of-the-art guardrails, e.g., LlamaGuard.



## **38. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2402.06255v4) [paper-pdf](http://arxiv.org/pdf/2402.06255v4)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreaking attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly focusing on model fine-tuning or heuristical defense designs. However, how to achieve intrinsic robustness through prompt optimization remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0%, while maintaining the model's utility on the benign task and incurring only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/PKU-ML/PAT.



## **39. Audio Is the Achilles' Heel: Red Teaming Audio Large Multimodal Models**

cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23861v1) [paper-pdf](http://arxiv.org/pdf/2410.23861v1)

**Authors**: Hao Yang, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari

**Abstract**: Large Multimodal Models (LMMs) have demonstrated the ability to interact with humans under real-world conditions by combining Large Language Models (LLMs) and modality encoders to align multimodal information (visual and auditory) with text. However, such models raise new safety challenges of whether models that are safety-aligned on text also exhibit consistent safeguards for multimodal inputs. Despite recent safety-alignment research on vision LMMs, the safety of audio LMMs remains under-explored. In this work, we comprehensively red team the safety of five advanced audio LMMs under three settings: (i) harmful questions in both audio and text formats, (ii) harmful questions in text format accompanied by distracting non-speech audio, and (iii) speech-specific jailbreaks. Our results under these settings demonstrate that open-source audio LMMs suffer an average attack success rate of 69.14% on harmful audio questions, and exhibit safety vulnerabilities when distracted with non-speech audio noise. Our speech-specific jailbreaks on Gemini-1.5-Pro achieve an attack success rate of 70.67% on the harmful query benchmark. We provide insights on what could cause these reported safety-misalignments. Warning: this paper contains offensive examples.



## **40. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Dataset & Benchmarking Track

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23746v1) [paper-pdf](http://arxiv.org/pdf/2410.23746v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **41. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23687v1) [paper-pdf](http://arxiv.org/pdf/2410.23687v1)

**Authors**: Chiyu Zhang, Xiaogang Xu, Jiafei Wu, Zhe Liu, Lu Zhou

**Abstract**: Adversarial attacks, which manipulate input data to undermine model availability and integrity, pose significant security threats during machine learning inference. With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreak techniques, have emerged. Understanding these attacks is crucial for developing more robust systems and demystifying the inner workings of neural networks. However, existing reviews often focus on attack classifications and lack comprehensive, in-depth analysis. The research community currently needs: 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations of existing methods; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.



## **42. Pseudo-Conversation Injection for LLM Goal Hijacking**

cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23678v1) [paper-pdf](http://arxiv.org/pdf/2410.23678v1)

**Authors**: Zheng Chen, Buhui Yao

**Abstract**: Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.



## **43. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

cs.SE

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2308.11161v2) [paper-pdf](http://arxiv.org/pdf/2308.11161v2)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.



## **44. Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**

cs.LG

Proceedings of the 7th BlackboxNLP Workshop: Analyzing and  Interpreting Neural Networks for NLP

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2401.10862v3) [paper-pdf](http://arxiv.org/pdf/2401.10862v3)

**Authors**: Adib Hasan, Ileana Rugina, Alex Wang

**Abstract**: This paper investigates the impact of model compression on the way Large Language Models (LLMs) process prompts, particularly concerning jailbreak resistance. We show that moderate WANDA pruning can enhance resistance to jailbreaking attacks without fine-tuning, while maintaining performance on standard benchmarks. To systematically evaluate this safety enhancement, we introduce a dataset of 225 harmful tasks across five categories. Our analysis of LLaMA-2 Chat, Vicuna 1.3, and Mistral Instruct v0.2 reveals that pruning benefits correlate with initial model safety levels. We interpret these results by examining changes in attention patterns and perplexity shifts, demonstrating that pruned models exhibit sharper attention and increased sensitivity to artificial jailbreak constructs. We extend our evaluation to the AdvBench harmful behavior tasks and the GCG attack method. We find that LLaMA-2 is much safer on AdvBench prompts than on our dataset when evaluated with manual jailbreak attempts, and that pruning is effective against both automated attacks and manual jailbreaking on Advbench.



## **45. Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23558v1) [paper-pdf](http://arxiv.org/pdf/2410.23558v1)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: In this report, we propose a novel black-box jailbreak attacking framework that incorporates various LLM-as-Attacker methods to deliver transferable and powerful jailbreak attacks. Our method is designed based on three key observations from existing jailbreaking studies and practices. First, we consider an ensemble approach should be more effective in exposing the vulnerabilities of an aligned LLM compared to individual attacks. Second, different malicious instructions inherently vary in their jailbreaking difficulty, necessitating differentiated treatment to ensure more efficient attacks. Finally, the semantic coherence of a malicious instruction is crucial for triggering the defenses of an aligned LLM; therefore, it must be carefully disrupted to manipulate its embedding representation, thereby increasing the jailbreak success rate. We validated our approach by participating in the Competition for LLM and Agent Safety 2024, where our team achieved top performance in the Jailbreaking Attack Track.



## **46. Representation Noising: A Defence Mechanism Against Harmful Finetuning**

cs.CL

Published in NeurIPs 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2405.14577v4) [paper-pdf](http://arxiv.org/pdf/2405.14577v4)

**Authors**: Domenic Rosati, Jan Wehner, Kai Williams, Łukasz Bartoszcze, David Atanasov, Robie Gonzales, Subhabrata Majumdar, Carsten Maple, Hassan Sajjad, Frank Rudzicz

**Abstract**: Releasing open-source large language models (LLMs) presents a dual-use risk since bad actors can easily fine-tune these models for harmful purposes. Even without the open release of weights, weight stealing and fine-tuning APIs make closed models vulnerable to harmful fine-tuning attacks (HFAs). While safety measures like preventing jailbreaks and improving safety guardrails are important, such measures can easily be reversed through fine-tuning. In this work, we propose Representation Noising (RepNoise), a defence mechanism that operates even when attackers have access to the weights. RepNoise works by removing information about harmful representations such that it is difficult to recover them during fine-tuning. Importantly, our defence is also able to generalize across different subsets of harm that have not been seen during the defence process as long as they are drawn from the same distribution of the attack set. Our method does not degrade the general capability of LLMs and retains the ability to train the model on harmless tasks. We provide empirical evidence that the efficacy of our defence lies in its ``depth'': the degree to which information about harmful representations is removed across all layers of the LLM. We also find areas where RepNoise still remains ineffective and highlight how those limitations can inform future research.



## **47. ProTransformer: Robustify Transformers via Plug-and-Play Paradigm**

cs.LG

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23182v1) [paper-pdf](http://arxiv.org/pdf/2410.23182v1)

**Authors**: Zhichao Hou, Weizhi Gao, Yuchen Shen, Feiyi Wang, Xiaorui Liu

**Abstract**: Transformer-based architectures have dominated various areas of machine learning in recent years. In this paper, we introduce a novel robust attention mechanism designed to enhance the resilience of transformer-based architectures. Crucially, this technique can be integrated into existing transformers as a plug-and-play layer, improving their robustness without the need for additional training or fine-tuning. Through comprehensive experiments and ablation studies, we demonstrate that our ProTransformer significantly enhances the robustness of transformer models across a variety of prediction tasks, attack mechanisms, backbone architectures, and data domains. Notably, without further fine-tuning, the ProTransformer consistently improves the performance of vanilla transformers by 19.5%, 28.3%, 16.1%, and 11.4% for BERT, ALBERT, DistilBERT, and RoBERTa, respectively, under the classical TextFooler attack. Furthermore, ProTransformer shows promising resilience in large language models (LLMs) against prompting-based attacks, improving the performance of T5 and LLaMA by 24.8% and 17.8%, respectively, and enhancing Vicuna by an average of 10.4% against the Jailbreaking attack. Beyond the language domain, ProTransformer also demonstrates outstanding robustness in both vision and graph domains.



## **48. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE



## **49. Stealth edits to large language models**

cs.AI

28 pages, 14 figures. Open source implementation:  https://github.com/qinghua-zhou/stealth-edits

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.12670v2) [paper-pdf](http://arxiv.org/pdf/2406.12670v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Wei Wang, Desmond J. Higham, Alexander N. Gorban, Alexander Bastounis, Ivan Y. Tyukin

**Abstract**: We reveal the theoretical foundations of techniques for editing large language models, and present new methods which can do so without requiring retraining. Our theoretical insights show that a single metric (a measure of the intrinsic dimension of the model's features) can be used to assess a model's editability and reveals its previously unrecognised susceptibility to malicious stealth attacks. This metric is fundamental to predicting the success of a variety of editing approaches, and reveals new bridges between disparate families of editing methods. We collectively refer to these as stealth editing methods, because they directly update a model's weights to specify its response to specific known hallucinating prompts without affecting other model behaviour. By carefully applying our theoretical insights, we are able to introduce a new jet-pack network block which is optimised for highly selective model editing, uses only standard network operations, and can be inserted into existing networks. We also reveal the vulnerability of language models to stealth attacks: a small change to a model's weights which fixes its response to a single attacker-chosen prompt. Stealth attacks are computationally simple, do not require access to or knowledge of the model's training data, and therefore represent a potent yet previously unrecognised threat to redistributed foundation models. Extensive experimental results illustrate and support our methods and their theoretical underpinnings. Demos and source code are available at https://github.com/qinghua-zhou/stealth-edits.



## **50. HijackRAG: Hijacking Attacks against Retrieval-Augmented Large Language Models**

cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22832v1) [paper-pdf](http://arxiv.org/pdf/2410.22832v1)

**Authors**: Yucheng Zhang, Qinfeng Li, Tianyu Du, Xuhong Zhang, Xinkui Zhao, Zhengwen Feng, Jianwei Yin

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge, making them adaptable and cost-effective for various applications. However, the growing reliance on these systems also introduces potential security risks. In this work, we reveal a novel vulnerability, the retrieval prompt hijack attack (HijackRAG), which enables attackers to manipulate the retrieval mechanisms of RAG systems by injecting malicious texts into the knowledge database. When the RAG system encounters target questions, it generates the attacker's pre-determined answers instead of the correct ones, undermining the integrity and trustworthiness of the system. We formalize HijackRAG as an optimization problem and propose both black-box and white-box attack strategies tailored to different levels of the attacker's knowledge. Extensive experiments on multiple benchmark datasets show that HijackRAG consistently achieves high attack success rates, outperforming existing baseline attacks. Furthermore, we demonstrate that the attack is transferable across different retriever models, underscoring the widespread risk it poses to RAG systems. Lastly, our exploration of various defense mechanisms reveals that they are insufficient to counter HijackRAG, emphasizing the urgent need for more robust security measures to protect RAG systems in real-world deployments.



