# Latest Adversarial Attack Papers
**update at 2025-02-24 09:53:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Red-Teaming LLM Multi-Agent Systems via Communication Attacks**

cs.CR

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14847v1) [paper-pdf](http://arxiv.org/pdf/2502.14847v1)

**Authors**: Pengfei He, Yupin Lin, Shen Dong, Han Xu, Yue Xing, Hui Liu

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.



## **2. Fundamental Limitations in Defending LLM Finetuning APIs**

cs.LG

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14828v1) [paper-pdf](http://arxiv.org/pdf/2502.14828v1)

**Authors**: Xander Davies, Eric Winsor, Tomek Korbak, Alexandra Souly, Robert Kirk, Christian Schroeder de Witt, Yarin Gal

**Abstract**: LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.



## **3. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

cs.CL

**SubmitDate**: 2025-02-21    [abs](http://arxiv.org/abs/2502.14744v2) [paper-pdf](http://arxiv.org/pdf/2502.14744v2)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.



## **4. SEA: Shareable and Explainable Attribution for Query-based Black-box Attacks**

cs.LG

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2308.11845v2) [paper-pdf](http://arxiv.org/pdf/2308.11845v2)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz

**Abstract**: Machine Learning (ML) systems are vulnerable to adversarial examples, particularly those from query-based black-box attacks. Despite various efforts to detect and prevent such attacks, ML systems are still at risk, demanding a more comprehensive approach to security that includes logging, analyzing, and sharing evidence. While traditional security benefits from well-established practices of forensics and threat intelligence sharing, ML security has yet to find a way to profile its attackers and share information about them. In response, this paper introduces SEA, a novel ML security system to characterize black-box attacks on ML systems for forensic purposes and to facilitate human-explainable intelligence sharing. SEA leverages Hidden Markov Models to attribute the observed query sequence to known attacks. It thus understands the attack's progression rather than focusing solely on the final adversarial examples. Our evaluations reveal that SEA is effective at attack attribution, even on the second incident, and is robust to adaptive strategies designed to evade forensic analysis. SEA's explanations of the attack's behavior allow us even to fingerprint specific minor bugs in widely used attack libraries. For example, we discover that the SignOPT and Square attacks in ART v1.14 send over 50% duplicated queries. We thoroughly evaluate SEA on a variety of settings and demonstrate that it can recognize the same attack with more than 90% Top-1 and 95% Top-3 accuracy. Finally, we demonstrate how SEA generalizes to other domains like text classification.



## **5. Pitch Imperfect: Detecting Audio Deepfakes Through Acoustic Prosodic Analysis**

cs.SD

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14726v1) [paper-pdf](http://arxiv.org/pdf/2502.14726v1)

**Authors**: Kevin Warren, Daniel Olszewski, Seth Layton, Kevin Butler, Carrie Gates, Patrick Traynor

**Abstract**: Audio deepfakes are increasingly in-differentiable from organic speech, often fooling both authentication systems and human listeners. While many techniques use low-level audio features or optimization black-box model training, focusing on the features that humans use to recognize speech will likely be a more long-term robust approach to detection. We explore the use of prosody, or the high-level linguistic features of human speech (e.g., pitch, intonation, jitter) as a more foundational means of detecting audio deepfakes. We develop a detector based on six classical prosodic features and demonstrate that our model performs as well as other baseline models used by the community to detect audio deepfakes with an accuracy of 93% and an EER of 24.7%. More importantly, we demonstrate the benefits of using a linguistic features-based approach over existing models by applying an adaptive adversary using an $L_{\infty}$ norm attack against the detectors and using attention mechanisms in our training for explainability. We show that we can explain the prosodic features that have highest impact on the model's decision (Jitter, Shimmer and Mean Fundamental Frequency) and that other models are extremely susceptible to simple $L_{\infty}$ norm attacks (99.3% relative degradation in accuracy). While overall performance may be similar, we illustrate the robustness and explainability benefits to a prosody feature approach to audio deepfake detection.



## **6. Moshi Moshi? A Model Selection Hijacking Adversarial Attack**

cs.LG

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14586v1) [paper-pdf](http://arxiv.org/pdf/2502.14586v1)

**Authors**: Riccardo Petrucci, Luca Pajola, Francesco Marchiori, Luca Pasa, Mauro conti

**Abstract**: Model selection is a fundamental task in Machine Learning~(ML), focusing on selecting the most suitable model from a pool of candidates by evaluating their performance on specific metrics. This process ensures optimal performance, computational efficiency, and adaptability to diverse tasks and environments. Despite its critical role, its security from the perspective of adversarial ML remains unexplored. This risk is heightened in the Machine-Learning-as-a-Service model, where users delegate the training phase and the model selection process to third-party providers, supplying data and training strategies. Therefore, attacks on model selection could harm both the user and the provider, undermining model performance and driving up operational costs.   In this work, we present MOSHI (MOdel Selection HIjacking adversarial attack), the first adversarial attack specifically targeting model selection. Our novel approach manipulates model selection data to favor the adversary, even without prior knowledge of the system. Utilizing a framework based on Variational Auto Encoders, we provide evidence that an attacker can induce inefficiencies in ML deployment. We test our attack on diverse computer vision and speech recognition benchmark tasks and different settings, obtaining an average attack success rate of 75.42%. In particular, our attack causes an average 88.30% decrease in generalization capabilities, an 83.33% increase in latency, and an increase of up to 105.85% in energy consumption. These results highlight the significant vulnerabilities in model selection processes and their potential impact on real-world applications.



## **7. Eliminating Backdoors in Neural Code Models for Secure Code Understanding**

cs.CR

Accepted to the 33rd ACM International Conference on the Foundations  of Software Engineering (FSE 2025)

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2408.04683v2) [paper-pdf](http://arxiv.org/pdf/2408.04683v2)

**Authors**: Weisong Sun, Yuchen Chen, Chunrong Fang, Yebo Feng, Yuan Xiao, An Guo, Quanjun Zhang, Yang Liu, Baowen Xu, Zhenyu Chen

**Abstract**: Neural code models (NCMs) have been widely used to address various code understanding tasks, such as defect detection. However, numerous recent studies reveal that such models are vulnerable to backdoor attacks. Backdoored NCMs function normally on normal/clean code snippets, but exhibit adversary-expected behavior on poisoned code snippets injected with the adversary-crafted trigger. It poses a significant security threat. Therefore, there is an urgent need for effective techniques to detect and eliminate backdoors stealthily implanted in NCMs.   To address this issue, in this paper, we innovatively propose a backdoor elimination technique for secure code understanding, called EliBadCode. EliBadCode eliminates backdoors in NCMs by inverting/reverse-engineering and unlearning backdoor triggers. Specifically, EliBadCode first filters the model vocabulary for trigger tokens based on the naming conventions of specific programming languages to reduce the trigger search space and cost. Then, EliBadCode introduces a sample-specific trigger position identification method, which can reduce the interference of non-backdoor (adversarial) perturbations for subsequent trigger inversion, thereby producing effective inverted backdoor triggers efficiently. Backdoor triggers can be viewed as backdoor (adversarial) perturbations. Subsequently, EliBadCode employs a Greedy Coordinate Gradient algorithm to optimize the inverted trigger and designs a trigger anchoring method to purify the inverted trigger. Finally, EliBadCode eliminates backdoors through model unlearning. We evaluate the effectiveness of EliBadCode in eliminating backdoors implanted in multiple NCMs used for three safety-critical code understanding tasks. The results demonstrate that EliBadCode can effectively eliminate backdoors while having minimal adverse effects on the normal functionality of the model.



## **8. Combined Quantum and Post-Quantum Security for Earth-Satellite Channels**

quant-ph

9 pages, 5 figures, 1 table, IEEE, International Conference on  Quantum Communications, Networking, and Computing 2025

**SubmitDate**: 2025-02-20    [abs](http://arxiv.org/abs/2502.14240v1) [paper-pdf](http://arxiv.org/pdf/2502.14240v1)

**Authors**: Anju Rani, Xiaoyu Ai, Aman Gupta, Ravi Singh Adhikari, Robert Malaney

**Abstract**: Experimental deployment of quantum communication over Earth-satellite channels opens the way to a secure global quantum Internet. In this work, we present results from a real-time prototype quantum key distribution (QKD) system, which entails the development of optical systems including the encoding of entangled photon pairs, the development of transmitters for quantum signaling through an emulated Earth-satellite channel, and the development of quantum-decoding receivers. A unique aspect of our system is the integration of QKD with existing cryptographic methods to ensure quantum-resistant security, even at low-key rates. In addition, we report the use of specially designed error-reconciliation codes that optimize the security versus key-rate trade-off. Our work demonstrates, for the first time, a deployment of the BBM92 protocol that offers both post-quantum security via the advanced encryption standard (AES) and quantum security via an entanglement-based QKD protocol. If either the AES or the QKD is compromised through some adversary attack, our system still delivers state-of-the-art communications secure against future quantum computers.



## **9. Scaling Trends in Language Model Robustness**

cs.LG

58 pages; updated to include new results and analysis

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2407.18213v4) [paper-pdf](http://arxiv.org/pdf/2407.18213v4)

**Authors**: Nikolaus Howe, Ian McKenzie, Oskar Hollinsworth, Michał Zajac, Tom Tseng, Aaron Tucker, Pierre-Luc Bacon, Adam Gleave

**Abstract**: Language models exhibit scaling laws, whereby increasing model and dataset size predictably decrease negative log likelihood, unlocking a dazzling array of capabilities. At the same time, even the most capable systems are currently vulnerable to adversarial inputs such as jailbreaks and prompt injections, despite concerted efforts to make them robust. As compute becomes more accessible to both attackers and defenders, which side will benefit more from scale? We attempt to answer this question with a detailed study of robustness on language models spanning three orders of magnitude in parameter count. From the defender's perspective, we find that in the absence of other interventions, increasing model size alone does not consistently improve robustness. In adversarial training, we find that larger models are more sample-efficient and less compute-efficient than smaller models, and often better generalize their defense to new threat models. From the attacker's perspective, we find that increasing attack compute smoothly and reliably increases attack success rate against both finetuned and adversarially trained models. Finally, we show that across model sizes studied, doubling compute on adversarial training only forces an attacker to less than double attack compute to maintain the same attack success rate. However, adversarial training becomes more and more effective on larger models, suggesting that defenders could eventually have the advantage with increasing model size. These results underscore the value of adopting a scaling lens when discussing robustness of frontier models.



## **10. Control Barrier Function based Attack-Recovery with Provable Guarantees**

eess.SY

V1: Conference version (IEEE CDC'2022) V2: Journal version (submitted  to IEEE Transactions on Automatic Control)

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2204.03077v2) [paper-pdf](http://arxiv.org/pdf/2204.03077v2)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstract**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack detection mechanism based on zeroing control barrier function (ZCBF) conditions. In addition, we design an adaptive recovery mechanism based on how close the system is to violating safety. We show that under certain conditions, the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. We propose sufficient conditions for the initial conditions and input constraints so that the resulting CPS is secure by design. We also propose a novel hybrid control to account for attack detection delays and avoid Zeno behavior. Next, to efficiently compute the set of initial conditions, we propose a sampling-based method to verify whether a set is a viability domain. Specifically, we devise a method for checking a modified barrier function condition on a finite set of points to assess whether a set can be rendered forward invariant. Then, we propose an iterative algorithm to compute the set of initial conditions and input constraints set to limit the effect of an adversary if it compromises vulnerable inputs. Finally, we use a Quadratic Programming (QP) approach for online recovery (as well as nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.



## **11. Carefully Blending Adversarial Training, Purification, and Aggregation Improves Adversarial Robustness**

cs.CV

25 pages, 1 figure, 16 tables

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2306.06081v5) [paper-pdf](http://arxiv.org/pdf/2306.06081v5)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and a carefully chosen aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for Cifar-10, Cifar-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at: https://github.com/emaballarin/CARSO .



## **12. Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach**

cs.CR

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2410.02890v4) [paper-pdf](http://arxiv.org/pdf/2410.02890v4)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.



## **13. Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives**

cs.SD

Accepted by ICLR 2025

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.11858v2) [paper-pdf](http://arxiv.org/pdf/2502.11858v2)

**Authors**: Zeliang Zhang, Susan Liang, Daiki Shimada, Chenliang Xu

**Abstract**: While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration also introduces new vulnerabilities to adversarial attacks.   In this paper, we present a comprehensive study of the adversarial robustness of audio-visual models, considering both temporal and modality-specific vulnerabilities. We propose two powerful adversarial attacks: 1) a temporal invariance attack that exploits the inherent temporal redundancy across consecutive time segments and 2) a modality misalignment attack that introduces incongruence between the audio and visual modalities. These attacks are designed to thoroughly assess the robustness of audio-visual models against diverse threats. Furthermore, to defend against such attacks, we introduce a novel audio-visual adversarial training framework. This framework addresses key challenges in vanilla adversarial training by incorporating efficient adversarial perturbation crafting tailored to multi-modal data and an adversarial curriculum strategy. Extensive experiments in the Kinetics-Sounds dataset demonstrate that our proposed temporal and modality-based attacks in degrading model performance can achieve state-of-the-art performance, while our adversarial training defense largely improves the adversarial robustness as well as the adversarial training efficiency.



## **14. Na'vi or Knave: Jailbreaking Language Models via Metaphorical Avatars**

cs.CL

We still need to polish our paper

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2412.12145v3) [paper-pdf](http://arxiv.org/pdf/2412.12145v3)

**Authors**: Yu Yan, Sheng Sun, Junqi Tong, Min Liu, Qi Li

**Abstract**: Metaphor serves as an implicit approach to convey information, while enabling the generalized comprehension of complex subjects. However, metaphor can potentially be exploited to bypass the safety alignment mechanisms of Large Language Models (LLMs), leading to the theft of harmful knowledge. In our study, we introduce a novel attack framework that exploits the imaginative capacity of LLMs to achieve jailbreaking, the J\underline{\textbf{A}}ilbreak \underline{\textbf{V}}ia \underline{\textbf{A}}dversarial Me\underline{\textbf{TA}} -pho\underline{\textbf{R}} (\textit{AVATAR}). Specifically, to elicit the harmful response, AVATAR extracts harmful entities from a given harmful target and maps them to innocuous adversarial entities based on LLM's imagination. Then, according to these metaphors, the harmful target is nested within human-like interaction for jailbreaking adaptively. Experimental results demonstrate that AVATAR can effectively and transferablly jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. Our study exposes a security risk in LLMs from their endogenous imaginative capabilities. Furthermore, the analytical study reveals the vulnerability of LLM to adversarial metaphors and the necessity of developing defense methods against jailbreaking caused by the adversarial metaphor. \textcolor{orange}{ \textbf{Warning: This paper contains potentially harmful content from LLMs.}}



## **15. Toward Robust Non-Transferable Learning: A Survey and Benchmark**

cs.LG

**SubmitDate**: 2025-02-19    [abs](http://arxiv.org/abs/2502.13593v1) [paper-pdf](http://arxiv.org/pdf/2502.13593v1)

**Authors**: Ziming Hong, Yongli Xiang, Tongliang Liu

**Abstract**: Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. In this paper, we bridge this gap by presenting the first comprehensive survey on NTL and introducing NTLBench, the first benchmark to evaluate NTL performance and robustness within a unified framework. Specifically, we first introduce the task settings, general framework, and criteria of NTL, followed by a summary of NTL approaches. Furthermore, we emphasize the often-overlooked issue of robustness against various attacks that can destroy the non-transferable mechanism established by NTL. Experiments conducted via NTLBench verify the limitations of existing NTL methods in robustness. Finally, we discuss the practical applications of NTL, along with its future directions and associated challenges.



## **16. Secure and Efficient Watermarking for Latent Diffusion Models in Model Distribution Scenarios**

cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13345v1) [paper-pdf](http://arxiv.org/pdf/2502.13345v1)

**Authors**: Liangqi Lei, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Latent diffusion models have exhibited considerable potential in generative tasks. Watermarking is considered to be an alternative to safeguard the copyright of generative models and prevent their misuse. However, in the context of model distribution scenarios, the accessibility of models to large scale of model users brings new challenges to the security, efficiency and robustness of existing watermark solutions. To address these issues, we propose a secure and efficient watermarking solution. A new security mechanism is designed to prevent watermark leakage and watermark escape, which considers watermark randomness and watermark-model association as two constraints for mandatory watermark injection. To reduce the time cost of training the security module, watermark injection and the security mechanism are decoupled, ensuring that fine-tuning VAE only accomplishes the security mechanism without the burden of learning watermark patterns. A watermark distribution-based verification strategy is proposed to enhance the robustness against diverse attacks in the model distribution scenarios. Experimental results prove that our watermarking consistently outperforms existing six baselines on effectiveness and robustness against ten image processing attacks and adversarial attacks, while enhancing security in the distribution scenarios.



## **17. UniGuardian: A Unified Defense for Detecting Prompt Injection, Backdoor Attacks and Adversarial Attacks in Large Language Models**

cs.CL

18 Pages, 8 Figures, 5 Tables, Keywords: Attack Defending, Security,  Prompt Injection, Backdoor Attacks, Adversarial Attacks, Prompt Trigger  Attacks

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13141v1) [paper-pdf](http://arxiv.org/pdf/2502.13141v1)

**Authors**: Huawei Lin, Yingjie Lao, Tong Geng, Tan Yu, Weijie Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to attacks like prompt injection, backdoor attacks, and adversarial attacks, which manipulate prompts or models to generate harmful outputs. In this paper, departing from traditional deep learning attack paradigms, we explore their intrinsic relationship and collectively term them Prompt Trigger Attacks (PTA). This raises a key question: Can we determine if a prompt is benign or poisoned? To address this, we propose UniGuardian, the first unified defense mechanism designed to detect prompt injection, backdoor attacks, and adversarial attacks in LLMs. Additionally, we introduce a single-forward strategy to optimize the detection pipeline, enabling simultaneous attack detection and text generation within a single forward pass. Our experiments confirm that UniGuardian accurately and efficiently identifies malicious prompts in LLMs.



## **18. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

cs.CL

18 pages

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12970v1) [paper-pdf](http://arxiv.org/pdf/2502.12970v1)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: The reasoning abilities of Large Language Models (LLMs) have demonstrated remarkable advancement and exceptional performance across diverse domains. However, leveraging these reasoning capabilities to enhance LLM safety against adversarial attacks and jailbreak queries remains largely unexplored. To bridge this gap, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates safety reflections of queries and responses into LLMs' generation process, unlocking a safety-aware reasoning mechanism. This approach enables self-evaluation at each reasoning step to create safety pivot tokens as indicators of the response's safety status. Furthermore, in order to improve the learning efficiency of pivot token prediction, we propose Contrastive Pivot Optimization(CPO), which enhances the model's ability to perceive the safety status of dialogues. Through this mechanism, LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their defense capabilities against jailbreak attacks. Extensive experimental results demonstrate that R2D effectively mitigates various attacks and improves overall safety, highlighting the substantial potential of safety-aware reasoning in strengthening LLMs' robustness against jailbreaks.



## **19. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13191v1) [paper-pdf](http://arxiv.org/pdf/2502.13191v1)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://anonymous.4open.science/r/MIA_SNN-3610.



## **20. Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training**

cs.CR

Submitted to ACL 2025, Preprint, Under review

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12734v1) [paper-pdf](http://arxiv.org/pdf/2502.12734v1)

**Authors**: Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu

**Abstract**: Machine-generated Text (MGT) detection is crucial for regulating and attributing online texts. While the existing MGT detectors achieve strong performance, they remain vulnerable to simple perturbations and adversarial attacks. To build an effective defense against malicious perturbations, we view MGT detection from a threat modeling perspective, that is, analyzing the model's vulnerability from an adversary's point of view and exploring effective mitigations. To this end, we introduce an adversarial framework for training a robust MGT detector, named GREedy Adversary PromoTed DefendER (GREATER). The GREATER consists of two key components: an adversary GREATER-A and a detector GREATER-D. The GREATER-D learns to defend against the adversarial attack from GREATER-A and generalizes the defense to other attacks. GREATER-A identifies and perturbs the critical tokens in embedding space, along with greedy search and pruning to generate stealthy and disruptive adversarial examples. Besides, we update the GREATER-A and GREATER-D synchronously, encouraging the GREATER-D to generalize its defense to different attacks and varying attack intensities. Our experimental results across 9 text perturbation strategies and 5 adversarial attacks show that our GREATER-D reduces the Attack Success Rate (ASR) by 10.61% compared with SOTA defense methods while our GREATER-A is demonstrated to be more effective and efficient than SOTA attack approaches.



## **21. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12659v1) [paper-pdf](http://arxiv.org/pdf/2502.12659v1)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **22. Chronus: Understanding and Securing the Cutting-Edge Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in HPCA'25. arXiv admin note: text overlap with  arXiv:2406.19094

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12650v1) [paper-pdf](http://arxiv.org/pdf/2502.12650v1)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Nisa Bostancı, İsmail Emir Yüksel, Haocong Luo, Oğuz Ergin, Onur Mutlu

**Abstract**: We 1) present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC) and 2) propose Chronus, a new mechanism that addresses PRAC's two major weaknesses. Our analysis shows that PRAC's system performance overhead on benign applications is non-negligible for modern DRAM chips and prohibitively large for future DRAM chips that are more vulnerable to read disturbance. We identify two weaknesses of PRAC that cause these overheads. First, PRAC increases critical DRAM access latency parameters due to the additional time required to increment activation counters. Second, PRAC performs a constant number of preventive refreshes at a time, making it vulnerable to an adversarial access pattern, known as the wave attack, and consequently requiring it to be configured for significantly smaller activation thresholds. To address PRAC's two weaknesses, we propose a new on-DRAM-die RowHammer mitigation mechanism, Chronus. Chronus 1) updates row activation counters concurrently while serving accesses by separating counters from the data and 2) prevents the wave attack by dynamically controlling the number of preventive refreshes performed. Our performance analysis shows that Chronus's system performance overhead is near-zero for modern DRAM chips and very low for future DRAM chips. Chronus outperforms three variants of PRAC and three other state-of-the-art read disturbance solutions. We discuss Chronus's and PRAC's implications for future systems and foreshadow future research directions. To aid future research, we open-source our Chronus implementation at https://github.com/CMU-SAFARI/Chronus.



## **23. Automating Prompt Leakage Attacks on Large Language Models Using Agentic Approach**

cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12630v1) [paper-pdf](http://arxiv.org/pdf/2502.12630v1)

**Authors**: Tvrtko Sternak, Davor Runje, Dorian Granoša, Chi Wang

**Abstract**: This paper presents a novel approach to evaluating the security of large language models (LLMs) against prompt leakage-the exposure of system-level prompts or proprietary configurations. We define prompt leakage as a critical threat to secure LLM deployment and introduce a framework for testing the robustness of LLMs using agentic teams. Leveraging AG2 (formerly AutoGen), we implement a multi-agent system where cooperative agents are tasked with probing and exploiting the target LLM to elicit its prompt.   Guided by traditional definitions of security in cryptography, we further define a prompt leakage-safe system as one in which an attacker cannot distinguish between two agents: one initialized with an original prompt and the other with a prompt stripped of all sensitive information. In a safe system, the agents' outputs will be indistinguishable to the attacker, ensuring that sensitive information remains secure. This cryptographically inspired framework provides a rigorous standard for evaluating and designing secure LLMs.   This work establishes a systematic methodology for adversarial testing of prompt leakage, bridging the gap between automated threat modeling and practical LLM security.   You can find the implementation of our prompt leakage probing on GitHub.



## **24. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

cs.CL

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2407.01461v2) [paper-pdf](http://arxiv.org/pdf/2407.01461v2)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .



## **25. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.13175v1) [paper-pdf](http://arxiv.org/pdf/2502.13175v1)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.



## **26. SoK: Understanding Vulnerabilities in the Large Language Model Supply Chain**

cs.CR

**SubmitDate**: 2025-02-18    [abs](http://arxiv.org/abs/2502.12497v1) [paper-pdf](http://arxiv.org/pdf/2502.12497v1)

**Authors**: Shenao Wang, Yanjie Zhao, Zhao Liu, Quanchen Zou, Haoyu Wang

**Abstract**: Large Language Models (LLMs) transform artificial intelligence, driving advancements in natural language understanding, text generation, and autonomous systems. The increasing complexity of their development and deployment introduces significant security challenges, particularly within the LLM supply chain. However, existing research primarily focuses on content safety, such as adversarial attacks, jailbreaking, and backdoor attacks, while overlooking security vulnerabilities in the underlying software systems. To address this gap, this study systematically analyzes 529 vulnerabilities reported across 75 prominent projects spanning 13 lifecycle stages. The findings show that vulnerabilities are concentrated in the application (50.3%) and model (42.7%) layers, with improper resource control (45.7%) and improper neutralization (25.1%) identified as the leading root causes. Additionally, while 56.7% of the vulnerabilities have available fixes, 8% of these patches are ineffective, resulting in recurring vulnerabilities. This study underscores the challenges of securing the LLM ecosystem and provides actionable insights to guide future research and mitigation strategies.



## **27. Independence Tests for Language Models**

cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.12292v1) [paper-pdf](http://arxiv.org/pdf/2502.12292v1)

**Authors**: Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, Percy Liang

**Abstract**: We consider the following problem: given the weights of two models, can we test whether they were trained independently -- i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose a family of statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. These p-values are valid regardless of the composition of either model's training data; we compute them by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures of weights and activations between the original two models versus these copies. We report the p-values from these tests on pairs of 21 open-weight models (210 total pairs) and correctly identify all pairs of non-independent models. Our tests remain effective even if one model was fine-tuned for many tokens. In the unconstrained setting, where we make no assumptions about training procedures, can change model architecture, and allow for adversarial evasion attacks, the previous tests no longer work. Instead, we propose a new test which matches hidden activations between two models, and which is robust to adversarial transformations and to changes in model architecture. The test can also do localized testing: identifying specific non-independent components of models. Though we no longer obtain exact p-values from this, empirically we find it behaves as one and reliably identifies non-independent models. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch.



## **28. Quantum Byzantine Multiple Access Channels**

cs.IT

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.12047v1) [paper-pdf](http://arxiv.org/pdf/2502.12047v1)

**Authors**: Minglai Cai, Christian Deppe

**Abstract**: In communication theory, attacks like eavesdropping or jamming are typically assumed to occur at the channel level, while communication parties are expected to follow established protocols. But what happens if one of the parties turns malicious? In this work, we investigate a compelling scenario: a multiple-access channel with two transmitters and one receiver, where one transmitter deviates from the protocol and acts dishonestly. To address this challenge, we introduce the Byzantine multiple-access classical-quantum channel and derive an achievable communication rate for this adversarial setting.



## **29. FedEAT: A Robustness Optimization Framework for Federated LLMs**

cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11863v1) [paper-pdf](http://arxiv.org/pdf/2502.11863v1)

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.



## **30. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

cs.CV

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2203.04607v3) [paper-pdf](http://arxiv.org/pdf/2203.04607v3)

**Authors**: Qilong Zhang, Youheng Sun, Chaoning Zhang, Chaoqun Li, Xuanhan Wang, Jingkuan Song, Lianli Gao

**Abstract**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.



## **31. Federated Multi-Armed Bandits Under Byzantine Attacks**

cs.LG

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2205.04134v3) [paper-pdf](http://arxiv.org/pdf/2205.04134v3)

**Authors**: Artun Saday, İlker Demirel, Yiğit Yıldırım, Cem Tekin

**Abstract**: Multi-armed bandits (MAB) is a sequential decision-making model in which the learner controls the trade-off between exploration and exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is an emerging framework where a cohort of learners with heterogeneous local models play an MAB game and communicate their aggregated feedback to a server to learn a globally optimal arm. Two key hurdles in FMAB are communication-efficient learning and resilience to adversarial attacks. To address these issues, we study the FMAB problem in the presence of Byzantine clients who can send false model updates threatening the learning process. We analyze the sample complexity and the regret of $\beta$-optimal arm identification. We borrow tools from robust statistics and propose a median-of-means (MoM)-based online algorithm, Fed-MoM-UCB, to cope with Byzantine clients. In particular, we show that if the Byzantine clients constitute less than half of the cohort, the cumulative regret with respect to $\beta$-optimal arms is bounded over time with high probability, showcasing both communication efficiency and Byzantine resilience. We analyze the interplay between the algorithm parameters, a discernibility margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.



## **32. Adversarially Robust CLIP Models Can Induce Better (Robust) Perceptual Metrics**

cs.CV

This work has been accepted for publication in the IEEE Conference on  Secure and Trustworthy Machine Learning (SaTML). The final version will be  available on IEEE Xplore

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11725v1) [paper-pdf](http://arxiv.org/pdf/2502.11725v1)

**Authors**: Francesco Croce, Christian Schlarmann, Naman Deep Singh, Matthias Hein

**Abstract**: Measuring perceptual similarity is a key tool in computer vision. In recent years perceptual metrics based on features extracted from neural networks with large and diverse training sets, e.g. CLIP, have become popular. At the same time, the metrics extracted from features of neural networks are not adversarially robust. In this paper we show that adversarially robust CLIP models, called R-CLIP$_\textrm{F}$, obtained by unsupervised adversarial fine-tuning induce a better and adversarially robust perceptual metric that outperforms existing metrics in a zero-shot setting, and further matches the performance of state-of-the-art metrics while being robust after fine-tuning. Moreover, our perceptual metric achieves strong performance on related tasks such as robust image-to-image retrieval, which becomes especially relevant when applied to "Not Safe for Work" (NSFW) content detection and dataset filtering. While standard perceptual metrics can be easily attacked by a small perturbation completely degrading NSFW detection, our robust perceptual metric maintains high accuracy under an attack while having similar performance for unperturbed images. Finally, perceptual metrics induced by robust CLIP models have higher interpretability: feature inversion can show which images are considered similar, while text inversion can find what images are associated to a given prompt. This also allows us to visualize the very rich visual concepts learned by a CLIP model, including memorized persons, paintings and complex queries.



## **33. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11647v1) [paper-pdf](http://arxiv.org/pdf/2502.11647v1)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.



## **34. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

cs.CL

22 pages, 12 figures, 13 tables

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11598v1) [paper-pdf](http://arxiv.org/pdf/2502.11598v1)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.



## **35. Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training**

cs.CR

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11455v1) [paper-pdf](http://arxiv.org/pdf/2502.11455v1)

**Authors**: Fenghua Weng, Jian Lou, Jun Feng, Minlie Huang, Wenjie Wang

**Abstract**: Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. Unlike LLM, the current safety alignment of VLMs is often achieved with post-hoc safety fine-tuning. However, these methods are less effective to white-box attacks. To address this, we propose $\textit{Adversary-aware DPO (ADPO)}$, a novel training framework that explicitly considers adversarial. $\textit{Adversary-aware DPO (ADPO)}$ integrates adversarial training into DPO to enhance the safety alignment of VLMs under worst-case adversarial perturbations. $\textit{ADPO}$ introduces two key components: (1) an adversarial-trained reference model that generates human-preferred responses under worst-case perturbations, and (2) an adversarial-aware DPO loss that generates winner-loser pairs accounting for adversarial distortions. By combining these innovations, $\textit{ADPO}$ ensures that VLMs remain robust and reliable even in the presence of sophisticated jailbreak attacks. Extensive experiments demonstrate that $\textit{ADPO}$ outperforms baselines in the safety alignment and general utility of VLMs.



## **36. Dagger Behind Smile: Fool LLMs with a Happy Ending Story**

cs.CL

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2501.13115v2) [paper-pdf](http://arxiv.org/pdf/2501.13115v2)

**Authors**: Xurui Song, Zhixin Xie, Shuo Huai, Jiayi Kong, Jun Luo

**Abstract**: The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.



## **37. Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System**

cs.AI

15 pages, 11 figures

**SubmitDate**: 2025-02-17    [abs](http://arxiv.org/abs/2502.11358v1) [paper-pdf](http://arxiv.org/pdf/2502.11358v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.



## **38. How to Backdoor Consistency Models?**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.19785v3) [paper-pdf](http://arxiv.org/pdf/2410.19785v3)

**Authors**: Chengen Wang, Murat Kantarcioglu

**Abstract**: Consistency models are a new class of models that generate images by directly mapping noise to data, allowing for one-step generation and significantly accelerating the sampling process. However, their robustness against adversarial attacks has not yet been thoroughly investigated. In this work, we conduct the first study on the vulnerability of consistency models to backdoor attacks. While previous research has explored backdoor attacks on diffusion models, those studies have primarily focused on conventional diffusion models, employing a customized backdoor training process and objective, whereas consistency models have distinct training processes and objectives. Our proposed framework demonstrates the vulnerability of consistency models to backdoor attacks. During image generation, poisoned consistency models produce images with a Fr\'echet Inception Distance (FID) comparable to that of a clean model when sampling from Gaussian noise. However, once the trigger is activated, they generate backdoor target images. We explore various trigger and target configurations to evaluate the vulnerability of consistency models, including the use of random noise as a trigger. This novel trigger is visually inconspicuous, more challenging to detect, and aligns well with the sampling process of consistency models. Across all configurations, our framework successfully compromises the consistency models while maintaining high utility and specificity. We also examine the stealthiness of our proposed attack, which is attributed to the unique properties of consistency models and the elusive nature of the Gaussian noise trigger. Our code is available at \href{https://github.com/chengenw/backdoorCM}{https://github.com/chengenw/backdoorCM}.



## **39. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

cs.LG

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.12207v1) [paper-pdf](http://arxiv.org/pdf/2502.12207v1)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://anonymous.4open.science/r/PAR-01BF/



## **40. ShieldLearner: A New Paradigm for Jailbreak Attack Defense in LLMs**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.13162v1) [paper-pdf](http://arxiv.org/pdf/2502.13162v1)

**Authors**: Ziyi Ni, Hao Wang, Huacan Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in various domains but remain vulnerable to adversarial jailbreak attacks. Existing prompt-defense strategies, including parameter-modifying and parameter-free approaches, face limitations in adaptability, interpretability, and customization, constraining their effectiveness against evolving threats. To address these challenges, we propose ShieldLearner, a novel paradigm that mimics human learning in defense. Through trial and error, it autonomously distills attack signatures into a Pattern Atlas and synthesizes defense heuristics into a Meta-analysis Framework, enabling systematic and interpretable threat detection. Furthermore, we introduce Adaptive Adversarial Augmentation to generate adversarial variations of successfully defended prompts, enabling continuous self-improvement without model retraining. In addition to standard benchmarks, we create a hard test set by curating adversarial prompts from the Wildjailbreak dataset, emphasizing more concealed malicious intent. Experimental results show that ShieldLearner achieves a significantly higher defense success rate than existing baselines on both conventional and hard test sets, while also operating with lower computational overhead, making it a practical and efficient solution for real-world adversarial defense.



## **41. G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11127v1) [paper-pdf](http://arxiv.org/pdf/2502.11127v1)

**Authors**: Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, Yang Wang

**Abstract**: Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborative problem-solving to autonomous decision-making. However, as these systems become increasingly integrated into critical applications, their vulnerability to adversarial attacks, misinformation propagation, and unintended behaviors have raised significant concerns. To address this challenge, we introduce G-Safeguard, a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. Extensive experiments demonstrate that G-Safeguard: (I) exhibits significant effectiveness under various attack strategies, recovering over 40% of the performance for prompt injection; (II) is highly adaptable to diverse LLM backbones and large-scale MAS; (III) can seamlessly combine with mainstream MAS with security guarantees. The code is available at https://github.com/wslong20/G-safeguard.



## **42. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2004.12571v4) [paper-pdf](http://arxiv.org/pdf/2004.12571v4)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.



## **43. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

cs.CL

21pages, 10 figures

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.11084v1) [paper-pdf](http://arxiv.org/pdf/2502.11084v1)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety.



## **44. BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack**

cs.CL

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2502.12202v1) [paper-pdf](http://arxiv.org/pdf/2502.12202v1)

**Authors**: Zihao Zhu, Hongbao Zhang, Mingda Zhang, Ruotong Wang, Guanzong Wu, Ke Xu, Baoyuan Wu

**Abstract**: Longer thought, better performance: large language models with deep reasoning capabilities, particularly o1-like models, have demonstrated remarkable performance by generating extensive thought processes during inference. This trade-off reveals a potential vulnerability: adversaries could compromise model performance by forcing immediate responses without thought processes. To this end, in this paper, we introduce a novel attack scenario targeting the long thought processes of o1-like models and propose BoT (Break CoT), which can selectively break intrinsic reasoning mechanisms through backdoor attacks. BoT constructs poisoned datasets with designed triggers and injects backdoor by either supervised fine-tuning or direct preference optimization. When triggered, the model directly generates answers without thought processes, while maintaining normal reasoning capabilities for clean inputs. Extensive experiments on open-source o1-like models, including recent DeepSeek-R1, demonstrate that BoT nearly achieves high attack success rates while maintaining clean accuracy, highlighting the critical safety risk in current models. Furthermore, the relationship between task difficulty and helpfulness reveals a potential application for good, enabling users to customize model behavior based on task complexity. Code is available at \href{https://github.com/zihao-ai/BoT}{https://github.com/zihao-ai/BoT}.



## **45. Atoxia: Red-teaming Large Language Models with Target Toxic Answers**

cs.CL

Accepted to Findings of NAACL-2025

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2408.14853v2) [paper-pdf](http://arxiv.org/pdf/2408.14853v2)

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Xiang Wan, Anningzhe Gao

**Abstract**: Despite the substantial advancements in artificial intelligence, large language models (LLMs) remain being challenged by generation safety. With adversarial jailbreaking prompts, one can effortlessly induce LLMs to output harmful content, causing unexpected negative social impacts. This vulnerability highlights the necessity for robust LLM red-teaming strategies to identify and mitigate such risks before large-scale application. To detect specific types of risks, we propose a novel red-teaming method that $\textbf{A}$ttacks LLMs with $\textbf{T}$arget $\textbf{Toxi}$c $\textbf{A}$nswers ($\textbf{Atoxia}$). Given a particular harmful answer, Atoxia generates a corresponding user query and a misleading answer opening to examine the internal defects of a given LLM. The proposed attacker is trained within a reinforcement learning scheme with the LLM outputting probability of the target answer as the reward. We verify the effectiveness of our method on various red-teaming benchmarks, such as AdvBench and HH-Harmless. The empirical results demonstrate that Atoxia can successfully detect safety risks in not only open-source models but also state-of-the-art black-box models such as GPT-4o.



## **46. JPEG Inspired Deep Learning**

cs.CV

**SubmitDate**: 2025-02-16    [abs](http://arxiv.org/abs/2410.07081v2) [paper-pdf](http://arxiv.org/pdf/2410.07081v2)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.



## **47. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

cs.CR

11 pages, 4 figures

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.07492v2) [paper-pdf](http://arxiv.org/pdf/2502.07492v2)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.



## **48. MITRE ATT&CK Applications in Cybersecurity and The Way Forward**

cs.CR

37 pages

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2502.10825v1) [paper-pdf](http://arxiv.org/pdf/2502.10825v1)

**Authors**: Yuning Jiang, Qiaoran Meng, Feiyang Shang, Nay Oo, Le Thi Hong Minh, Hoon Wei Lim, Biplab Sikdar

**Abstract**: The MITRE ATT&CK framework is a widely adopted tool for enhancing cybersecurity, supporting threat intelligence, incident response, attack modeling, and vulnerability prioritization. This paper synthesizes research on its application across these domains by analyzing 417 peer-reviewed publications. We identify commonly used adversarial tactics, techniques, and procedures (TTPs) and examine the integration of natural language processing (NLP) and machine learning (ML) with ATT&CK to improve threat detection and response. Additionally, we explore the interoperability of ATT&CK with other frameworks, such as the Cyber Kill Chain, NIST guidelines, and STRIDE, highlighting its versatility. The paper further evaluates the framework from multiple perspectives, including its effectiveness, validation methods, and sector-specific challenges, particularly in industrial control systems (ICS) and healthcare. We conclude by discussing current limitations and proposing future research directions to enhance the applicability of ATT&CK in dynamic cybersecurity environments.



## **49. Pixel Is Not a Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models**

cs.CV

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2408.11810v3) [paper-pdf](http://arxiv.org/pdf/2408.11810v3)

**Authors**: Chun-Yen Shih, Li-Xuan Peng, Jia-Wei Liao, Ernie Chu, Cheng-Fu Chou, Jun-Cheng Chen

**Abstract**: Diffusion Models have emerged as powerful generative models for high-quality image synthesis, with many subsequent image editing techniques based on them. However, the ease of text-based image editing introduces significant risks, such as malicious editing for scams or intellectual property infringement. Previous works have attempted to safeguard images from diffusion-based editing by adding imperceptible perturbations. These methods are costly and specifically target prevalent Latent Diffusion Models (LDMs), while Pixel-domain Diffusion Models (PDMs) remain largely unexplored and robust against such attacks. Our work addresses this gap by proposing a novel attack framework, AtkPDM. AtkPDM is mainly composed of a feature representation attacking loss that exploits vulnerabilities in denoising UNets and a latent optimization strategy to enhance the naturalness of adversarial images. Extensive experiments demonstrate the effectiveness of our approach in attacking dominant PDM-based editing methods (e.g., SDEdit) while maintaining reasonable fidelity and robustness against common defense methods. Additionally, our framework is extensible to LDMs, achieving comparable performance to existing approaches.



## **50. Robustness-aware Automatic Prompt Optimization**

cs.CL

**SubmitDate**: 2025-02-15    [abs](http://arxiv.org/abs/2412.18196v2) [paper-pdf](http://arxiv.org/pdf/2412.18196v2)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Hang Gao, Fan Yang, Ruixiang Tang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) depends on the quality of prompts and the semantic and structural integrity of the input data. However, existing prompt generation methods primarily focus on well-structured input data, often neglecting the impact of perturbed inputs on prompt effectiveness. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt does not need access to model parameters and gradients. Instead, BATprompt leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. We evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.



