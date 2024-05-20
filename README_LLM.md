# Latest Large Language Model Attack Papers
**update at 2024-05-20 09:24:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. A Comprehensive Study of Jailbreak Attack versus Defense for Large Language Models**

cs.CR

18 pages, 9 figures, Accepted in ACL 2024

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2402.13457v2) [paper-pdf](http://arxiv.org/pdf/2402.13457v2)

**Authors**: Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, Stjepan Picek

**Abstract**: Large Language Models (LLMS) have increasingly become central to generating content with potential societal impacts. Notably, these models have demonstrated capabilities for generating content that could be deemed harmful. To mitigate these risks, researchers have adopted safety training techniques to align model outputs with societal values to curb the generation of malicious content. However, the phenomenon of "jailbreaking", where carefully crafted prompts elicit harmful responses from models, persists as a significant challenge. This research conducts a comprehensive analysis of existing studies on jailbreaking LLMs and their defense techniques. We meticulously investigate nine attack techniques and seven defense techniques applied across three distinct language models: Vicuna, LLama, and GPT-3.5 Turbo. We aim to evaluate the effectiveness of these attack and defense techniques. Our findings reveal that existing white-box attacks underperform compared to universal techniques and that including special tokens in the input significantly affects the likelihood of successful attacks. This research highlights the need to concentrate on the security facets of LLMs. Additionally, we contribute to the field by releasing our datasets and testing framework, aiming to foster further research into LLM security. We believe these contributions will facilitate the exploration of security measures within this domain.



## **2. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

cs.CV

15 pages

**SubmitDate**: 2024-05-17    [abs](http://arxiv.org/abs/2405.10529v1) [paper-pdf](http://arxiv.org/pdf/2405.10529v1)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.



## **3. Keep It Private: Unsupervised Privatization of Online Text**

cs.CL

17 pages, 6 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.10260v1) [paper-pdf](http://arxiv.org/pdf/2405.10260v1)

**Authors**: Calvin Bao, Marine Carpuat

**Abstract**: Authorship obfuscation techniques hold the promise of helping people protect their privacy in online communications by automatically rewriting text to hide the identity of the original author. However, obfuscation has been evaluated in narrow settings in the NLP literature and has primarily been addressed with superficial edit operations that can lead to unnatural outputs. In this work, we introduce an automatic text privatization framework that fine-tunes a large language model via reinforcement learning to produce rewrites that balance soundness, sense, and privacy. We evaluate it extensively on a large-scale test set of English Reddit posts by 68k authors composed of short-medium length texts. We study how the performance changes among evaluative conditions including authorial profile length and authorship detection strategy. Our method maintains high text quality according to both automated metrics and human evaluation, and successfully evades several automated authorship attacks.



## **4. Protecting Your LLMs with Information Bottleneck**

cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **5. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.



## **6. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2403.01218v2) [paper-pdf](http://arxiv.org/pdf/2403.01218v2)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **7. Transfer Learning in Pre-Trained Large Language Models for Malware Detection Based on System Calls**

cs.CR

Submitted to IEEE MILCOM 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09318v1) [paper-pdf](http://arxiv.org/pdf/2405.09318v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Gregorio Martínez Pérez

**Abstract**: In the current cybersecurity landscape, protecting military devices such as communication and battlefield management systems against sophisticated cyber attacks is crucial. Malware exploits vulnerabilities through stealth methods, often evading traditional detection mechanisms such as software signatures. The application of ML/DL in vulnerability detection has been extensively explored in the literature. However, current ML/DL vulnerability detection methods struggle with understanding the context and intent behind complex attacks. Integrating large language models (LLMs) with system call analysis offers a promising approach to enhance malware detection. This work presents a novel framework leveraging LLMs to classify malware based on system call data. The framework uses transfer learning to adapt pre-trained LLMs for malware detection. By retraining LLMs on a dataset of benign and malicious system calls, the models are refined to detect signs of malware activity. Experiments with a dataset of over 1TB of system calls demonstrate that models with larger context sizes, such as BigBird and Longformer, achieve superior accuracy and F1-Score of approximately 0.86. The results highlight the importance of context size in improving detection rates and underscore the trade-offs between computational complexity and performance. This approach shows significant potential for real-time detection in high-stakes environments, offering a robust solution to evolving cyber threats.



## **8. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

cs.CR

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2308.03825v2) [paper-pdf](http://arxiv.org/pdf/2308.03825v2)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has drawn significant attention from the general public and LLM vendors. One particular type of adversarial prompt, known as jailbreak prompt, has emerged as the main attack vector to bypass the safeguards and elicit harmful content from LLMs. In this paper, employing our new framework JailbreakHub, we conduct a comprehensive analysis of 1,405 jailbreak prompts spanning from December 2022 to December 2023. We identify 131 jailbreak communities and discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from online Web communities to prompt-aggregation websites and 28 user accounts have consistently optimized jailbreak prompts over 100 days. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 107,250 samples across 13 forbidden scenarios. Leveraging this dataset, our experiments on six popular LLMs show that their safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify five highly effective jailbreak prompts that achieve 0.95 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and the earliest one has persisted online for over 240 days. We hope that our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.



## **9. Large Language Models can be Guided to Evade AI-Generated Text Detection**

cs.CL

TMLR camera ready

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2305.10847v6) [paper-pdf](http://arxiv.org/pdf/2305.10847v6)

**Authors**: Ning Lu, Shengcai Liu, Rui He, Qi Wang, Yew-Soon Ong, Ke Tang

**Abstract**: Large language models (LLMs) have shown remarkable performance in various tasks and have been extensively utilized by the public. However, the increasing concerns regarding the misuse of LLMs, such as plagiarism and spamming, have led to the development of multiple detectors, including fine-tuned classifiers and statistical methods. In this study, we equip LLMs with prompts, rather than relying on an external paraphraser, to evaluate the vulnerability of these detectors. We propose a novel Substitution-based In-Context example Optimization method (SICO) to automatically construct prompts for evading the detectors. SICO is cost-efficient as it requires only 40 human-written examples and a limited number of LLM inferences to generate a prompt. Moreover, once a task-specific prompt has been constructed, it can be universally used against a wide range of detectors. Extensive experiments across three real-world tasks demonstrate that SICO significantly outperforms the paraphraser baselines and enables GPT-3.5 to successfully evade six detectors, decreasing their AUC by 0.5 on average. Furthermore, a comprehensive human evaluation show that the SICO-generated text achieves human-level readability and task completion rates, while preserving high imperceptibility. Finally, we propose an ensemble approach to enhance the robustness of detectors against SICO attack. The code is publicly available at https://github.com/ColinLu50/Evade-GPT-Detector.



## **10. Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization**

cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09113v1) [paper-pdf](http://arxiv.org/pdf/2405.09113v1)

**Authors**: Kai Hu, Weichen Yu, Tianjun Yao, Xiang Li, Wenhe Liu, Lijun Yu, Yining Li, Kai Chen, Zhiqiang Shen, Matt Fredrikson

**Abstract**: Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which effectively jailbreaks several open-source LLMs. Our approach relaxes the discrete jailbreak optimization into a continuous optimization and progressively increases the sparsity of the optimizing vectors. Consequently, our method effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than existing token-level methods. On Harmbench, our method achieves state of the art attack success rate on seven out of eight LLMs. Code will be made available. Trigger Warning: This paper contains model behavior that can be offensive in nature.



## **11. A safety realignment framework via subspace-oriented model fusion for large language models**

cs.CL

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09055v1) [paper-pdf](http://arxiv.org/pdf/2405.09055v1)

**Authors**: Xin Yi, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: The current safeguard mechanisms for large language models (LLMs) are indeed susceptible to jailbreak attacks, making them inherently fragile. Even the process of fine-tuning on apparently benign data for downstream tasks can jeopardize safety. One potential solution is to conduct safety fine-tuning subsequent to downstream fine-tuning. However, there's a risk of catastrophic forgetting during safety fine-tuning, where LLMs may regain safety measures but lose the task-specific knowledge acquired during downstream fine-tuning. In this paper, we introduce a safety realignment framework through subspace-oriented model fusion (SOMF), aiming to combine the safeguard capabilities of initially aligned model and the current fine-tuned model into a realigned model. Our approach begins by disentangling all task vectors from the weights of each fine-tuned model. We then identify safety-related regions within these vectors by subspace masking techniques. Finally, we explore the fusion of the initial safely aligned LLM with all task vectors based on the identified safety subspace. We validate that our safety realignment framework satisfies the safety requirements of a single fine-tuned model as well as multiple models during their fusion. Our findings confirm that SOMF preserves safety without notably compromising performance on downstream tasks, including instruction following in Chinese, English, and Hindi, as well as problem-solving capabilities in Code and Math.



## **12. Distributed Threat Intelligence at the Edge Devices: A Large Language Model-Driven Approach**

cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08755v1) [paper-pdf](http://arxiv.org/pdf/2405.08755v1)

**Authors**: Syed Mhamudul Hasan, Alaa M. Alotaibi, Sajedul Talukder, Abdur R. Shahid

**Abstract**: With the proliferation of edge devices, there is a significant increase in attack surface on these devices. The decentralized deployment of threat intelligence on edge devices, coupled with adaptive machine learning techniques such as the in-context learning feature of large language models (LLMs), represents a promising paradigm for enhancing cybersecurity on low-powered edge devices. This approach involves the deployment of lightweight machine learning models directly onto edge devices to analyze local data streams, such as network traffic and system logs, in real-time. Additionally, distributing computational tasks to an edge server reduces latency and improves responsiveness while also enhancing privacy by processing sensitive data locally. LLM servers can enable these edge servers to autonomously adapt to evolving threats and attack patterns, continuously updating their models to improve detection accuracy and reduce false positives. Furthermore, collaborative learning mechanisms facilitate peer-to-peer secure and trustworthy knowledge sharing among edge devices, enhancing the collective intelligence of the network and enabling dynamic threat mitigation measures such as device quarantine in response to detected anomalies. The scalability and flexibility of this approach make it well-suited for diverse and evolving network environments, as edge devices only send suspicious information such as network traffic and system log changes, offering a resilient and efficient solution to combat emerging cyber threats at the network edge. Thus, our proposed framework can improve edge computing security by providing better security in cyber threat detection and mitigation by isolating the edge devices from the network.



## **13. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.



## **14. Stylometric Watermarks for Large Language Models**

cs.CL

19 pages, 4 figures, 9 tables

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08400v1) [paper-pdf](http://arxiv.org/pdf/2405.08400v1)

**Authors**: Georg Niess, Roman Kern

**Abstract**: The rapid advancement of large language models (LLMs) has made it increasingly difficult to distinguish between text written by humans and machines. Addressing this, we propose a novel method for generating watermarks that strategically alters token probabilities during generation. Unlike previous works, this method uniquely employs linguistic features such as stylometry. Concretely, we introduce acrostica and sensorimotor norms to LLMs. Further, these features are parameterized by a key, which is updated every sentence. To compute this key, we use semantic zero shot classification, which enhances resilience. In our evaluation, we find that for three or more sentences, our method achieves a false positive and false negative rate of 0.02. For the case of a cyclic translation attack, we observe similar results for seven or more sentences. This research is of particular of interest for proprietary LLMs to facilitate accountability and prevent societal harm.



## **15. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.



## **16. Many-Shot Regurgitation (MSR) Prompting**

cs.CL

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.08134v1) [paper-pdf](http://arxiv.org/pdf/2405.08134v1)

**Authors**: Shashank Sonkar, Richard G. Baraniuk

**Abstract**: We introduce Many-Shot Regurgitation (MSR) prompting, a new black-box membership inference attack framework for examining verbatim content reproduction in large language models (LLMs). MSR prompting involves dividing the input text into multiple segments and creating a single prompt that includes a series of faux conversation rounds between a user and a language model to elicit verbatim regurgitation. We apply MSR prompting to diverse text sources, including Wikipedia articles and open educational resources (OER) textbooks, which provide high-quality, factual content and are continuously updated over time. For each source, we curate two dataset types: one that LLMs were likely exposed to during training ($D_{\rm pre}$) and another consisting of documents published after the models' training cutoff dates ($D_{\rm post}$). To quantify the occurrence of verbatim matches, we employ the Longest Common Substring algorithm and count the frequency of matches at different length thresholds. We then use statistical measures such as Cliff's delta, Kolmogorov-Smirnov (KS) distance, and Kruskal-Wallis H test to determine whether the distribution of verbatim matches differs significantly between $D_{\rm pre}$ and $D_{\rm post}$. Our findings reveal a striking difference in the distribution of verbatim matches between $D_{\rm pre}$ and $D_{\rm post}$, with the frequency of verbatim reproduction being significantly higher when LLMs (e.g. GPT models and LLaMAs) are prompted with text from datasets they were likely trained on. For instance, when using GPT-3.5 on Wikipedia articles, we observe a substantial effect size (Cliff's delta $= -0.984$) and a large KS distance ($0.875$) between the distributions of $D_{\rm pre}$ and $D_{\rm post}$. Our results provide compelling evidence that LLMs are more prone to reproducing verbatim content when the input text is likely sourced from their training data.



## **17. Backdoor Removal for Generative Large Language Models**

cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.



## **18. DoLLM: How Large Language Models Understanding Network Flow Data to Detect Carpet Bombing DDoS**

cs.NI

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07638v1) [paper-pdf](http://arxiv.org/pdf/2405.07638v1)

**Authors**: Qingyang Li, Yihang Zhang, Zhidong Jia, Yannan Hu, Lei Zhang, Jianrong Zhang, Yongming Xu, Yong Cui, Zongming Guo, Xinggong Zhang

**Abstract**: It is an interesting question Can and How Large Language Models (LLMs) understand non-language network data, and help us detect unknown malicious flows. This paper takes Carpet Bombing as a case study and shows how to exploit LLMs' powerful capability in the networking area. Carpet Bombing is a new DDoS attack that has dramatically increased in recent years, significantly threatening network infrastructures. It targets multiple victim IPs within subnets, causing congestion on access links and disrupting network services for a vast number of users. Characterized by low-rates, multi-vectors, these attacks challenge traditional DDoS defenses. We propose DoLLM, a DDoS detection model utilizes open-source LLMs as backbone. By reorganizing non-contextual network flows into Flow-Sequences and projecting them into LLMs semantic space as token embeddings, DoLLM leverages LLMs' contextual understanding to extract flow representations in overall network context. The representations are used to improve the DDoS detection performance. We evaluate DoLLM with public datasets CIC-DDoS2019 and real NetFlow trace from Top-3 countrywide ISP. The tests have proven that DoLLM possesses strong detection capabilities. Its F1 score increased by up to 33.3% in zero-shot scenarios and by at least 20.6% in real ISP traces.



## **19. ExplainableDetector: Exploring Transformer-based Language Modeling Approach for SMS Spam Detection with Explainability Analysis**

cs.LG

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2405.08026v1) [paper-pdf](http://arxiv.org/pdf/2405.08026v1)

**Authors**: Mohammad Amaz Uddin, Muhammad Nazrul Islam, Leandros Maglaras, Helge Janicke, Iqbal H. Sarker

**Abstract**: SMS, or short messaging service, is a widely used and cost-effective communication medium that has sadly turned into a haven for unwanted messages, commonly known as SMS spam. With the rapid adoption of smartphones and Internet connectivity, SMS spam has emerged as a prevalent threat. Spammers have taken notice of the significance of SMS for mobile phone users. Consequently, with the emergence of new cybersecurity threats, the number of SMS spam has expanded significantly in recent years. The unstructured format of SMS data creates significant challenges for SMS spam detection, making it more difficult to successfully fight spam attacks in the cybersecurity domain. In this work, we employ optimized and fine-tuned transformer-based Large Language Models (LLMs) to solve the problem of spam message detection. We use a benchmark SMS spam dataset for this spam detection and utilize several preprocessing techniques to get clean and noise-free data and solve the class imbalance problem using the text augmentation technique. The overall experiment showed that our optimized fine-tuned BERT (Bidirectional Encoder Representations from Transformers) variant model RoBERTa obtained high accuracy with 99.84\%. We also work with Explainable Artificial Intelligence (XAI) techniques to calculate the positive and negative coefficient scores which explore and explain the fine-tuned model transparency in this text-based spam SMS detection task. In addition, traditional Machine Learning (ML) models were also examined to compare their performance with the transformer-based models. This analysis describes how LLMs can make a good impact on complex textual-based spam data in the cybersecurity field.



## **20. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.



## **21. LLMs and the Future of Chip Design: Unveiling Security Risks and Building Trust**

cs.LG

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2405.07061v1) [paper-pdf](http://arxiv.org/pdf/2405.07061v1)

**Authors**: Zeng Wang, Lilas Alrahis, Likhitha Mankali, Johann Knechtel, Ozgur Sinanoglu

**Abstract**: Chip design is about to be revolutionized by the integration of large language, multimodal, and circuit models (collectively LxMs). While exploring this exciting frontier with tremendous potential, the community must also carefully consider the related security risks and the need for building trust into using LxMs for chip design. First, we review the recent surge of using LxMs for chip design in general. We cover state-of-the-art works for the automation of hardware description language code generation and for scripting and guidance of essential but cumbersome tasks for electronic design automation tools, e.g., design-space exploration, tuning, or designer training. Second, we raise and provide initial answers to novel research questions on critical issues for security and trustworthiness of LxM-powered chip design from both the attack and defense perspectives.



## **22. Talk Too Much: Poisoning Large Language Models under Token Limit**

cs.CL

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.14795v3) [paper-pdf](http://arxiv.org/pdf/2404.14795v3)

**Authors**: Jiaming He, Wenbo Jiang, Guanyu Hou, Wenshu Fan, Rui Zhang, Hongwei Li

**Abstract**: Mainstream poisoning attacks on large language models (LLMs) typically set a fixed trigger in the input instance and specific responses for triggered queries. However, the fixed trigger setting (e.g., unusual words) may be easily detected by human detection, limiting the effectiveness and practicality in real-world scenarios. To enhance the stealthiness of the trigger, we present a poisoning attack against LLMs that is triggered by a generation/output condition-token limitation, which is a commonly adopted strategy by users for reducing costs. The poisoned model performs normally for output without token limitation, while becomes harmful for output with limited tokens. To achieve this objective, we introduce BrieFool, an efficient attack framework. It leverages the characteristics of generation limitation by efficient instruction sampling and poisoning data generation, thereby influencing the behavior of LLMs under target conditions. Our experiments demonstrate that BrieFool is effective across safety domains and knowledge domains. For instance, with only 20 generated poisoning examples against GPT-3.5-turbo, BrieFool achieves a 100% Attack Success Rate (ASR) and a 9.28/10 average Harmfulness Score (HS) under token limitation conditions while maintaining the benign performance.



## **23. Explaining Arguments' Strength: Unveiling the Role of Attacks and Supports (Technical Report)**

cs.AI

This paper has been accepted at IJCAI 2024 (the 33rd International  Joint Conference on Artificial Intelligence)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2404.14304v2) [paper-pdf](http://arxiv.org/pdf/2404.14304v2)

**Authors**: Xiang Yin, Potyka Nico, Francesca Toni

**Abstract**: Quantitatively explaining the strength of arguments under gradual semantics has recently received increasing attention. Specifically, several works in the literature provide quantitative explanations by computing the attribution scores of arguments. These works disregard the importance of attacks and supports, even though they play an essential role when explaining arguments' strength. In this paper, we propose a novel theory of Relation Attribution Explanations (RAEs), adapting Shapley values from game theory to offer fine-grained insights into the role of attacks and supports in quantitative bipolar argumentation towards obtaining the arguments' strength. We show that RAEs satisfy several desirable properties. We also propose a probabilistic algorithm to approximate RAEs efficiently. Finally, we show the application value of RAEs in fraud detection and large language models case studies.



## **24. Risks of Practicing Large Language Models in Smart Grid: Threat Modeling and Validation**

cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06237v1) [paper-pdf](http://arxiv.org/pdf/2405.06237v1)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Sun

**Abstract**: Large Language Model (LLM) is a significant breakthrough in artificial intelligence (AI) and holds considerable potential for application within smart grids. However, as demonstrated in previous literature, AI technologies are susceptible to various types of attacks. It is crucial to investigate and evaluate the risks associated with LLMs before deploying them in critical infrastructure like smart grids. In this paper, we systematically evaluate the vulnerabilities of LLMs and identify two major types of attacks relevant to smart grid LLM applications, along with presenting the corresponding threat models. We then validate these attacks using popular LLMs, utilizing real smart grid data. Our validation demonstrates that attackers are capable of injecting bad data and retrieving domain knowledge from LLMs employed in smart grid scenarios.



## **25. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.



## **26. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.



## **27. LLMPot: Automated LLM-based Industrial Protocol and Physical Process Emulation for ICS Honeypots**

cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05999v1) [paper-pdf](http://arxiv.org/pdf/2405.05999v1)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.



## **28. Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for LLM**

cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05610v1) [paper-pdf](http://arxiv.org/pdf/2405.05610v1)

**Authors**: Xikang Yang, Xuehai Tang, Songlin Hu, Jizhong Han

**Abstract**: Large language models (LLMs) have achieved remarkable performance in various natural language processing tasks, especially in dialogue systems. However, LLM may also pose security and moral threats, especially in multi round conversations where large models are more easily guided by contextual content, resulting in harmful or biased responses. In this paper, we present a novel method to attack LLMs in multi-turn dialogues, called CoA (Chain of Attack). CoA is a semantic-driven contextual multi-turn attack method that adaptively adjusts the attack policy through contextual feedback and semantic relevance during multi-turn of dialogue with a large model, resulting in the model producing unreasonable or harmful content. We evaluate CoA on different LLMs and datasets, and show that it can effectively expose the vulnerabilities of LLMs, and outperform existing attack methods. Our work provides a new perspective and tool for attacking and defending LLMs, and contributes to the security and ethical assessment of dialogue systems.



## **29. Large Language Models for Cyber Security: A Systematic Literature Review**

cs.CR

46 pages,6 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.04760v2) [paper-pdf](http://arxiv.org/pdf/2405.04760v2)

**Authors**: HanXiang Xu, ShenAo Wang, NingKe Li, KaiLong Wang, YanJie Zhao, Kai Chen, Ting Yu, Yang Liu, HaoYu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.



## **30. Special Characters Attack: Toward Scalable Training Data Extraction From Large Language Models**

cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05990v1) [paper-pdf](http://arxiv.org/pdf/2405.05990v1)

**Authors**: Yang Bai, Ge Pei, Jindong Gu, Yong Yang, Xingjun Ma

**Abstract**: Large language models (LLMs) have achieved remarkable performance on a wide range of tasks. However, recent studies have shown that LLMs can memorize training data and simple repeated tokens can trick the model to leak the data. In this paper, we take a step further and show that certain special characters or their combinations with English letters are stronger memory triggers, leading to more severe data leakage. The intuition is that, since LLMs are trained with massive data that contains a substantial amount of special characters (e.g. structural symbols {, } of JSON files, and @, # in emails and online posts), the model may memorize the co-occurrence between these special characters and the raw texts. This motivates us to propose a simple but effective Special Characters Attack (SCA) to induce training data leakage. Our experiments verify the high effectiveness of SCA against state-of-the-art LLMs: they can leak diverse training data, such as code corpus, web pages, and personally identifiable information, and sometimes generate non-stop outputs as a byproduct. We further show that the composition of the training data corpus can be revealed by inspecting the leaked data -- one crucial piece of information for pre-training high-performance LLMs. Our work can help understand the sensitivity of LLMs to special characters and identify potential areas for improvement.



## **31. Locally Differentially Private In-Context Learning**

cs.CR

This paper was published at LREC-Coling 2024

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04032v2) [paper-pdf](http://arxiv.org/pdf/2405.04032v2)

**Authors**: Chunyan Zheng, Keke Sun, Wenhao Zhao, Haibo Zhou, Lixin Jiang, Shaoyang Song, Chunlai Zhou

**Abstract**: Large pretrained language models (LLMs) have shown surprising In-Context Learning (ICL) ability. An important application in deploying large language models is to augment LLMs with a private database for some specific task. The main problem with this promising commercial use is that LLMs have been shown to memorize their training data and their prompt data are vulnerable to membership inference attacks (MIA) and prompt leaking attacks. In order to deal with this problem, we treat LLMs as untrusted in privacy and propose a locally differentially private framework of in-context learning(LDP-ICL) in the settings where labels are sensitive. Considering the mechanisms of in-context learning in Transformers by gradient descent, we provide an analysis of the trade-off between privacy and utility in such LDP-ICL for classification. Moreover, we apply LDP-ICL to the discrete distribution estimation problem. In the end, we perform several experiments to demonstrate our analysis results.



## **32. Air Gap: Protecting Privacy-Conscious Conversational Agents**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.



## **33. Critical Infrastructure Protection: Generative AI, Challenges, and Opportunities**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04874v1) [paper-pdf](http://arxiv.org/pdf/2405.04874v1)

**Authors**: Yagmur Yigit, Mohamed Amine Ferrag, Iqbal H. Sarker, Leandros A. Maglaras, Christos Chrysoulas, Naghmeh Moradpoor, Helge Janicke

**Abstract**: Critical National Infrastructure (CNI) encompasses a nation's essential assets that are fundamental to the operation of society and the economy, ensuring the provision of vital utilities such as energy, water, transportation, and communication. Nevertheless, growing cybersecurity threats targeting these infrastructures can potentially interfere with operations and seriously risk national security and public safety. In this paper, we examine the intricate issues raised by cybersecurity risks to vital infrastructure, highlighting these systems' vulnerability to different types of cyberattacks. We analyse the significance of trust, privacy, and resilience for Critical Infrastructure Protection (CIP), examining the diverse standards and regulations to manage these domains. We also scrutinise the co-analysis of safety and security, offering innovative approaches for their integration and emphasising the interdependence between these fields. Furthermore, we introduce a comprehensive method for CIP leveraging Generative AI and Large Language Models (LLMs), giving a tailored lifecycle and discussing specific applications across different critical infrastructure sectors. Lastly, we discuss potential future directions that promise to enhance the security and resilience of critical infrastructures. This paper proposes innovative strategies for CIP from evolving attacks and enhances comprehension of cybersecurity concerns related to critical infrastructure.



## **34. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.



## **35. AttacKG+:Boosting Attack Knowledge Graph Construction with Large Language Models**

cs.CR

20 pages, 5 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04753v1) [paper-pdf](http://arxiv.org/pdf/2405.04753v1)

**Authors**: Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang Wang, Yi Xie, Guozheng Yang, Yuliang Lu, Ee-Chien Chang

**Abstract**: Attack knowledge graph construction seeks to convert textual cyber threat intelligence (CTI) reports into structured representations, portraying the evolutionary traces of cyber attacks. Even though previous research has proposed various methods to construct attack knowledge graphs, they generally suffer from limited generalization capability to diverse knowledge types as well as requirement of expertise in model design and tuning. Addressing these limitations, we seek to utilize Large Language Models (LLMs), which have achieved enormous success in a broad range of tasks given exceptional capabilities in both language understanding and zero-shot task fulfillment. Thus, we propose a fully automatic LLM-based framework to construct attack knowledge graphs named: AttacKG+. Our framework consists of four consecutive modules: rewriter, parser, identifier, and summarizer, each of which is implemented by instruction prompting and in-context learning empowered by LLMs. Furthermore, we upgrade the existing attack knowledge schema and propose a comprehensive version. We represent a cyber attack as a temporally unfolding event, each temporal step of which encapsulates three layers of representation, including behavior graph, MITRE TTP labels, and state summary. Extensive evaluation demonstrates that: 1) our formulation seamlessly satisfies the information needs in threat event analysis, 2) our construction framework is effective in faithfully and accurately extracting the information defined by AttacKG+, and 3) our attack graph directly benefits downstream security practices such as attack reconstruction. All the code and datasets will be released upon acceptance.



## **36. Revisiting character-level adversarial attacks**

cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.



## **37. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.



## **38. Are aligned neural networks adversarially aligned?**

cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.



## **39. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.



## **40. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.



## **41. Confidential and Protected Disease Classifier using Fully Homomorphic Encryption**

cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02790v1) [paper-pdf](http://arxiv.org/pdf/2405.02790v1)

**Authors**: Aditya Malik, Nalini Ratha, Bharat Yalavarthi, Tilak Sharma, Arjun Kaushik, Charanjit Jutla

**Abstract**: With the rapid surge in the prevalence of Large Language Models (LLMs), individuals are increasingly turning to conversational AI for initial insights across various domains, including health-related inquiries such as disease diagnosis. Many users seek potential causes on platforms like ChatGPT or Bard before consulting a medical professional for their ailment. These platforms offer valuable benefits by streamlining the diagnosis process, alleviating the significant workload of healthcare practitioners, and saving users both time and money by avoiding unnecessary doctor visits. However, Despite the convenience of such platforms, sharing personal medical data online poses risks, including the presence of malicious platforms or potential eavesdropping by attackers. To address privacy concerns, we propose a novel framework combining FHE and Deep Learning for a secure and private diagnosis system. Operating on a question-and-answer-based model akin to an interaction with a medical practitioner, this end-to-end secure system employs Fully Homomorphic Encryption (FHE) to handle encrypted input data. Given FHE's computational constraints, we adapt deep neural networks and activation functions to the encryted domain. Further, we also propose a faster algorithm to compute summation of ciphertext elements. Through rigorous experiments, we demonstrate the efficacy of our approach. The proposed framework achieves strict security and privacy with minimal loss in performance.



## **42. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.



## **43. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

cs.LG

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2310.06387v2) [paper-pdf](http://arxiv.org/pdf/2310.06387v2)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating harmful content have emerged. In this paper, we delve into the potential of In-Context Learning (ICL) to modulate the alignment of LLMs. Specifically, we propose the In-Context Attack (ICA), which employs strategically crafted harmful demonstrations to subvert LLMs, and the In-Context Defense (ICD), which bolsters model resilience through examples that demonstrate refusal to produce harmful responses. Through extensive experiments, we demonstrate the efficacy of ICA and ICD in respectively elevating and mitigating the success rates of jailbreaking prompts. Moreover, we offer theoretical insights into the mechanism by which a limited set of in-context demonstrations can pivotally influence the safety alignment of LLMs. Our findings illuminate the profound influence of ICL on LLM behavior, opening new avenues for improving the safety and alignment of LLMs.



## **44. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation**

cs.SE

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02580v1) [paper-pdf](http://arxiv.org/pdf/2405.02580v1)

**Authors**: Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu

**Abstract**: With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs, such as GPT-4, to transfer existing human-written properties (e.g., those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new prop- erty for a given code. While this basic process is relatively straight- forward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) runtime-verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated prop- erties. We have implemented these strategies into a novel system called PropertyGPT, with 623 human-written properties collected from 23 Certora projects. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, resulting in $8,256 bug bounty rewards.



## **45. Adaptive and robust watermark against model extraction attack**

cs.CR

**SubmitDate**: 2024-05-03    [abs](http://arxiv.org/abs/2405.02365v1) [paper-pdf](http://arxiv.org/pdf/2405.02365v1)

**Authors**: Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai

**Abstract**: Large language models have boosted Large Models as a Service (LMaaS) into a thriving business sector. But even model owners offering only API access while keeping model parameters and internal workings private, their Intellectual Property (IP) are still at risk of theft through model extraction attacks. To safeguard the IP of these models and mitigate unfair competition in the language model market, watermarking technology serves as an efficient post-hoc solution for identifying IP infringements. However, existing IP protection watermarking methods either explicitly alter the original output of the language model or implant watermark signals in the model logits. These methods forcefully distort the original distribution of the language model and impact the sampling process, leading to a decline in the quality of the generated text. The existing method also fails to achieve end-to-end adaptive watermark embedding and lack robustness verification in complex scenarios where watermark detection is subject to interference. To overcome these challenges, we propose PromptShield, a plug-and-play IP protection watermarking method to resist model extraction attacks without training additional modules. Leveraging the self-reminding properties inherent in large language models, we encapsulate the user's query with a watermark self-generated instruction, nudging the LLMs to automatically generate watermark words in its output without compromising generation quality. Our method does not require access to the model's internal logits and minimizes alterations to the model's distribution using prompt-guided cues. Comprehensive experimental results consistently demonstrate the effectiveness, harmlessness, and robustness of our watermark. Moreover, Our watermark detection method remains robust and high detection sensitivity even when subjected to interference.



## **46. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2312.03853v3) [paper-pdf](http://arxiv.org/pdf/2312.03853v3)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then followed a role-play style to elicit prohibited responses. By making use of personas, we show that such responses are actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, which show that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **47. Generative AI in Cybersecurity**

cs.CR

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01674v1) [paper-pdf](http://arxiv.org/pdf/2405.01674v1)

**Authors**: Shivani Metta, Isaac Chang, Jack Parker, Michael P. Roman, Arturo F. Ehuan

**Abstract**: The dawn of Generative Artificial Intelligence (GAI), characterized by advanced models such as Generative Pre-trained Transformers (GPT) and other Large Language Models (LLMs), has been pivotal in reshaping the field of data analysis, pattern recognition, and decision-making processes. This surge in GAI technology has ushered in not only innovative opportunities for data processing and automation but has also introduced significant cybersecurity challenges.   As GAI rapidly progresses, it outstrips the current pace of cybersecurity protocols and regulatory frameworks, leading to a paradox wherein the same innovations meant to safeguard digital infrastructures also enhance the arsenal available to cyber criminals. These adversaries, adept at swiftly integrating and exploiting emerging technologies, may utilize GAI to develop malware that is both more covert and adaptable, thus complicating traditional cybersecurity efforts.   The acceleration of GAI presents an ambiguous frontier for cybersecurity experts, offering potent tools for threat detection and response, while concurrently providing cyber attackers with the means to engineer more intricate and potent malware. Through the joint efforts of Duke Pratt School of Engineering, Coalfire, and Safebreach, this research undertakes a meticulous analysis of how malicious agents are exploiting GAI to augment their attack strategies, emphasizing a critical issue for the integrity of future cybersecurity initiatives. The study highlights the critical need for organizations to proactively identify and develop more complex defensive strategies to counter the sophisticated employment of GAI in malware creation.



## **48. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2308.07308v4) [paper-pdf](http://arxiv.org/pdf/2308.07308v4)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense



## **49. Boosting Jailbreak Attack with Momentum**

cs.LG

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.01229v1) [paper-pdf](http://arxiv.org/pdf/2405.01229v1)

**Authors**: Yihao Zhang, Zeming Wei

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse tasks, yet they remain vulnerable to adversarial attacks, notably the well-documented \textit{jailbreak} attack. Recently, the Greedy Coordinate Gradient (GCG) attack has demonstrated efficacy in exploiting this vulnerability by optimizing adversarial prompts through a combination of gradient heuristics and greedy search. However, the efficiency of this attack has become a bottleneck in the attacking process. To mitigate this limitation, in this paper we rethink the generation of adversarial prompts through an optimization lens, aiming to stabilize the optimization process and harness more heuristic insights from previous iterations. Specifically, we introduce the \textbf{M}omentum \textbf{A}ccelerated G\textbf{C}G (\textbf{MAC}) attack, which incorporates a momentum term into the gradient heuristic. Experimental results showcase the notable enhancement achieved by MAP in gradient-based attacks on aligned language models. Our code is available at https://github.com/weizeming/momentum-attack-llm.



## **50. Adversarial Attacks and Defense for Conversation Entailment Task**

cs.CL

**SubmitDate**: 2024-05-02    [abs](http://arxiv.org/abs/2405.00289v2) [paper-pdf](http://arxiv.org/pdf/2405.00289v2)

**Authors**: Zhenning Yang, Ryan Krawec, Liang-Yuan Wu

**Abstract**: As the deployment of NLP systems in critical applications grows, ensuring the robustness of large language models (LLMs) against adversarial attacks becomes increasingly important. Large language models excel in various NLP tasks but remain vulnerable to low-cost adversarial attacks. Focusing on the domain of conversation entailment, where multi-turn dialogues serve as premises to verify hypotheses, we fine-tune a transformer model to accurately discern the truthfulness of these hypotheses. Adversaries manipulate hypotheses through synonym swapping, aiming to deceive the model into making incorrect predictions. To counteract these attacks, we implemented innovative fine-tuning techniques and introduced an embedding perturbation loss method to significantly bolster the model's robustness. Our findings not only emphasize the importance of defending against adversarial attacks in NLP but also highlight the real-world implications, suggesting that enhancing model robustness is critical for reliable NLP applications.



