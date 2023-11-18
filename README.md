# Latest Adversarial Attack Papers
**update at 2023-11-18 10:50:43**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2309.06978v2) [paper-pdf](http://arxiv.org/pdf/2309.06978v2)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **2. Towards more Practical Threat Models in Artificial Intelligence Security**

cs.CR

18 pages, 4 figures, 7 tables, under submission

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09994v1) [paper-pdf](http://arxiv.org/pdf/2311.09994v1)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with \textbf{271} industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.



## **3. Hijacking Large Language Models via Adversarial In-Context Learning**

cs.LG

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09948v1) [paper-pdf](http://arxiv.org/pdf/2311.09948v1)

**Authors**: Yao Qiang, Xiangyu Zhou, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific tasks by utilizing labeled examples as demonstrations in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable attack for ICL, aiming to hijack LLMs to generate the targeted response. The proposed LLM hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demonstrations. Extensive experimental results on various tasks and datasets demonstrate the effectiveness of our LLM hijacking attack, resulting in a distracted attention towards adversarial tokens, consequently leading to the targeted unwanted outputs.



## **4. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-start**

stat.ML

Corrected Remark 18 + other small edits. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2202.03397v4) [paper-pdf](http://arxiv.org/pdf/2202.03397v4)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstract**: We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e.~they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.



## **5. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

cs.LG

12 pages, 2 figures, 5 tables

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09790v1) [paper-pdf](http://arxiv.org/pdf/2311.09790v1)

**Authors**: Ilbert Romain, V. Hoang Thai, Zhang Zonghua, Palpanas Themis

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. % In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. % Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. % Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. % Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.



## **6. On the Exploitability of Reinforcement Learning with Human Feedback for Large Language Models**

cs.AI

**SubmitDate**: 2023-11-16    [abs](http://arxiv.org/abs/2311.09641v1) [paper-pdf](http://arxiv.org/pdf/2311.09641v1)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.



## **7. HAL 9000: Skynet's Risk Manager**

cs.CR

18 pages, 9 figures

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09449v1) [paper-pdf](http://arxiv.org/pdf/2311.09449v1)

**Authors**: Tadeu Freitas, Mário Neto, Inês Dutra, João Soares, Manuel Correia, Rolando Martins

**Abstract**: Intrusion Tolerant Systems (ITSs) are a necessary component for cyber-services/infrastructures. Additionally, as cyberattacks follow a multi-domain attack surface, a similar defensive approach should be applied, namely, the use of an evolving multi-disciplinary solution that combines ITS, cybersecurity and Artificial Intelligence (AI). With the increased popularity of AI solutions, due to Big Data use-case scenarios and decision support and automation scenarios, new opportunities to apply Machine Learning (ML) algorithms have emerged, namely ITS empowerment. Using ML algorithms, an ITS can augment its intrusion tolerance capability, by learning from previous attacks and from known vulnerabilities. As such, this work's contribution is twofold: (1) an ITS architecture (Skynet) based on the state-of-the-art and incorporates new components to increase its intrusion tolerance capability and its adaptability to new adversaries; (2) an improved Risk Manager design that leverages AI to improve ITSs by automatically assessing OS risks to intrusions, and advise with safer configurations. One of the reasons that intrusions are successful is due to bad configurations or slow adaptability to new threats. This can be caused by the dependency that systems have for human intervention. One of the characteristics in Skynet and HAL 9000 design is the removal of human intervention. Being fully automatized lowers the chance of successful intrusions caused by human error. Our experiments using Skynet, shows that HAL is able to choose 15% safer configurations than the state-of-the-art risk manager.



## **8. How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities**

cs.CL

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09447v1) [paper-pdf](http://arxiv.org/pdf/2311.09447v1)

**Authors**: Lingbo Mo, Boshi Wang, Muhao Chen, Huan Sun

**Abstract**: The rapid progress in open-source Large Language Models (LLMs) is significantly driving AI development forward. However, there is still a limited understanding of their trustworthiness. Deploying these models at scale without sufficient trustworthiness can pose significant risks, highlighting the need to uncover these issues promptly. In this work, we conduct an assessment of open-source LLMs on trustworthiness, scrutinizing them across eight different aspects including toxicity, stereotypes, ethics, hallucination, fairness, sycophancy, privacy, and robustness against adversarial demonstrations. We propose an enhanced Chain of Utterances-based (CoU) prompting strategy by incorporating meticulously crafted malicious demonstrations for trustworthiness attack. Our extensive experiments encompass recent and representative series of open-source LLMs, including Vicuna, MPT, Falcon, Mistral, and Llama 2. The empirical outcomes underscore the efficacy of our attack strategy across diverse aspects. More interestingly, our result analysis reveals that models with superior performance in general NLP tasks do not always have greater trustworthiness; in fact, larger models can be more vulnerable to attacks. Additionally, models that have undergone instruction tuning, focusing on instruction following, tend to be more susceptible, although fine-tuning LLMs for safety alignment proves effective in mitigating adversarial trustworthiness attacks.



## **9. Beyond Detection: Unveiling Fairness Vulnerabilities in Abusive Language Models**

cs.CL

Under review

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09428v1) [paper-pdf](http://arxiv.org/pdf/2311.09428v1)

**Authors**: Yueqing Liang, Lu Cheng, Ali Payani, Kai Shu

**Abstract**: This work investigates the potential of undermining both fairness and detection performance in abusive language detection. In a dynamic and complex digital world, it is crucial to investigate the vulnerabilities of these detection models to adversarial fairness attacks to improve their fairness robustness. We propose a simple yet effective framework FABLE that leverages backdoor attacks as they allow targeted control over the fairness and detection performance. FABLE explores three types of trigger designs (i.e., rare, artificial, and natural triggers) and novel sampling strategies. Specifically, the adversary can inject triggers into samples in the minority group with the favored outcome (i.e., ``non-abusive'') and flip their labels to the unfavored outcome, i.e., ``abusive''. Experiments on benchmark datasets demonstrate the effectiveness of FABLE attacking fairness and utility in abusive language detection.



## **10. UMD: Unsupervised Model Detection for X2X Backdoor Attacks**

cs.LG

Proceedings of the 40th International Conference on Machine Learning

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2305.18651v4) [paper-pdf](http://arxiv.org/pdf/2305.18651v4)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.



## **11. Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD**

cs.LG

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2307.00310v2) [paper-pdf](http://arxiv.org/pdf/2307.00310v2)

**Authors**: Anvith Thudi, Hengrui Jia, Casey Meehan, Ilia Shumailov, Nicolas Papernot

**Abstract**: Differentially private stochastic gradient descent (DP-SGD) is the canonical approach to private deep learning. While the current privacy analysis of DP-SGD is known to be tight in some settings, several empirical results suggest that models trained on common benchmark datasets leak significantly less privacy for many datapoints. Yet, despite past attempts, a rigorous explanation for why this is the case has not been reached. Is it because there exist tighter privacy upper bounds when restricted to these dataset settings, or are our attacks not strong enough for certain datapoints? In this paper, we provide the first per-instance (i.e., ``data-dependent") DP analysis of DP-SGD. Our analysis captures the intuition that points with similar neighbors in the dataset enjoy better data-dependent privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints (when trained on common benchmarks) than the current data-independent guarantee. This implies privacy attacks will necessarily fail against many datapoints if the adversary does not have sufficient control over the possible training datasets.



## **12. Frontier Language Models are not Robust to Adversarial Arithmetic, or "What do I need to say so you agree 2+2=5?**

cs.CL

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.07587v2) [paper-pdf](http://arxiv.org/pdf/2311.07587v2)

**Authors**: C. Daniel Freeman, Laura Culp, Aaron Parisi, Maxwell L Bileschi, Gamaleldin F Elsayed, Alex Rizkowsky, Isabelle Simpson, Alex Alemi, Azade Nova, Ben Adlam, Bernd Bohnet, Gaurav Mishra, Hanie Sedghi, Igor Mordatch, Izzeddin Gur, Jaehoon Lee, JD Co-Reyes, Jeffrey Pennington, Kelvin Xu, Kevin Swersky, Kshiteej Mahajan, Lechao Xiao, Rosanne Liu, Simon Kornblith, Noah Constant, Peter J. Liu, Roman Novak, Yundi Qian, Noah Fiedel, Jascha Sohl-Dickstein

**Abstract**: We introduce and study the problem of adversarial arithmetic, which provides a simple yet challenging testbed for language model alignment. This problem is comprised of arithmetic questions posed in natural language, with an arbitrary adversarial string inserted before the question is complete. Even in the simple setting of 1-digit addition problems, it is easy to find adversarial prompts that make all tested models (including PaLM2, GPT4, Claude2) misbehave, and even to steer models to a particular wrong answer. We additionally provide a simple algorithm for finding successful attacks by querying those same models, which we name "prompt inversion rejection sampling" (PIRS). We finally show that models can be partially hardened against these attacks via reinforcement learning and via agentic constitutional loops. However, we were not able to make a language model fully robust against adversarial arithmetic attacks.



## **13. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

cs.CR

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09127v1) [paper-pdf](http://arxiv.org/pdf/2311.09127v1)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities in model APIs. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully steal the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2)Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking, which could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.



## **14. Fast Certification of Vision-Language Models Using Incremental Randomized Smoothing**

cs.CV

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09024v1) [paper-pdf](http://arxiv.org/pdf/2311.09024v1)

**Authors**: A K Nirala, A Joshi, C Hegde, S Sarkar

**Abstract**: A key benefit of deep vision-language models such as CLIP is that they enable zero-shot open vocabulary classification; the user has the ability to define novel class labels via natural language prompts at inference time. However, while CLIP-based zero-shot classifiers have demonstrated competitive performance across a range of domain shifts, they remain highly vulnerable to adversarial attacks. Therefore, ensuring the robustness of such models is crucial for their reliable deployment in the wild.   In this work, we introduce Open Vocabulary Certification (OVC), a fast certification method designed for open-vocabulary models like CLIP via randomized smoothing techniques. Given a base "training" set of prompts and their corresponding certified CLIP classifiers, OVC relies on the observation that a classifier with a novel prompt can be viewed as a perturbed version of nearby classifiers in the base training set. Therefore, OVC can rapidly certify the novel classifier using a variation of incremental randomized smoothing. By using a caching trick, we achieve approximately two orders of magnitude acceleration in the certification process for novel prompts. To achieve further (heuristic) speedups, OVC approximates the embedding space at a given input using a multivariate normal distribution bypassing the need for sampling via forward passes through the vision backbone. We demonstrate the effectiveness of OVC on through experimental evaluation using multiple vision-language backbones on the CIFAR-10 and ImageNet test datasets.



## **15. Adversarial Attacks to Reward Machine-based Reinforcement Learning**

cs.LG

Thesis Supervisor: Prof. Federico Cerutti (Universit\`a degli Studi  di Brescia, IT)

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2311.09014v1) [paper-pdf](http://arxiv.org/pdf/2311.09014v1)

**Authors**: Lorenzo Nodari

**Abstract**: In recent years, Reward Machines (RMs) have stood out as a simple yet effective automata-based formalism for exposing and exploiting task structure in reinforcement learning settings. Despite their relevance, little to no attention has been directed to the study of their security implications and robustness to adversarial scenarios, likely due to their recent appearance in the literature. With my thesis, I aim to provide the first analysis of the security of RM-based reinforcement learning techniques, with the hope of motivating further research in the field, and I propose and evaluate a novel class of attacks on RM-based techniques: blinding attacks.



## **16. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

cs.LG

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2301.12554v3) [paper-pdf](http://arxiv.org/pdf/2301.12554v3)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that build neural classifiers robust against adversarial robustness, practitioners are still reluctant to adopt them due to their unacceptably severe clean accuracy penalties. This paper significantly alleviates this accuracy-robustness trade-off by mixing the output probabilities of a standard classifier and a robust classifier, where the standard network is optimized for clean accuracy and is not robust in general. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key to this improvement. In addition to providing intuitions and empirical evidence, we theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon = 8/255$) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.



## **17. On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**

stat.ML

16 pages, 3 figures

**SubmitDate**: 2023-11-15    [abs](http://arxiv.org/abs/2310.14421v4) [paper-pdf](http://arxiv.org/pdf/2310.14421v4)

**Authors**: Illia Horenko

**Abstract**: Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.



## **18. DALA: A Distribution-Aware LoRA-Based Adversarial Attack against Pre-trained Language Models**

cs.CL

First two authors contribute equally

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08598v1) [paper-pdf](http://arxiv.org/pdf/2311.08598v1)

**Authors**: Yibo Wang, Xiangjue Dong, James Caverlee, Philip S. Yu

**Abstract**: Pre-trained language models (PLMs) that achieve success in applications are susceptible to adversarial attack methods that are capable of generating adversarial examples with minor perturbations. Although recent attack methods can achieve a relatively high attack success rate (ASR), our observation shows that the generated adversarial examples have a different data distribution compared with the original examples. Specifically, these adversarial examples exhibit lower confidence levels and higher distance to the training data distribution. As a result, they are easy to detect using very simple detection methods, diminishing the actual effectiveness of these attack methods. To solve this problem, we propose a Distribution-Aware LoRA-based Adversarial Attack (DALA) method, which considers the distribution shift of adversarial examples to improve attack effectiveness under detection methods. We further design a new evaluation metric NASR combining ASR and detection for the attack task. We conduct experiments on four widely-used datasets and validate the attack effectiveness on ASR and NASR of the adversarial examples generated by DALA on the BERT-base model and the black-box LLaMA2-7b model.



## **19. Physical Adversarial Examples for Multi-Camera Systems**

cs.CV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08539v1) [paper-pdf](http://arxiv.org/pdf/2311.08539v1)

**Authors**: Ana Răduţoiu, Jan-Philipp Schulze, Philip Sperl, Konstantin Böttinger

**Abstract**: Neural networks build the foundation of several intelligent systems, which, however, are known to be easily fooled by adversarial examples. Recent advances made these attacks possible even in air-gapped scenarios, where the autonomous system observes its surroundings by, e.g., a camera. We extend these ideas in our research and evaluate the robustness of multi-camera setups against such physical adversarial examples. This scenario becomes ever more important with the rise in popularity of autonomous vehicles, which fuse the information of several cameras for their driving decision. While we find that multi-camera setups provide some robustness towards past attack methods, we see that this advantage reduces when optimizing on multiple perspectives at once. We propose a novel attack method that we call Transcender-MC, where we incorporate online 3D renderings and perspective projections in the training process. Moreover, we motivate that certain data augmentation techniques can facilitate the generation of successful adversarial examples even further. Transcender-MC is 11% more effective in successfully attacking multi-camera setups than state-of-the-art methods. Our findings offer valuable insights regarding the resilience of object detection in a setup with multiple cameras and motivate the need of developing adequate defense mechanisms against them.



## **20. Alignment is not sufficient to prevent large language models from generating harmful information: A psychoanalytic perspective**

cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08487v1) [paper-pdf](http://arxiv.org/pdf/2311.08487v1)

**Authors**: Zi Yin, Wei Ding, Jia Liu

**Abstract**: Large Language Models (LLMs) are central to a multitude of applications but struggle with significant risks, notably in generating harmful content and biases. Drawing an analogy to the human psyche's conflict between evolutionary survival instincts and societal norm adherence elucidated in Freud's psychoanalysis theory, we argue that LLMs suffer a similar fundamental conflict, arising between their inherent desire for syntactic and semantic continuity, established during the pre-training phase, and the post-training alignment with human values. This conflict renders LLMs vulnerable to adversarial attacks, wherein intensifying the models' desire for continuity can circumvent alignment efforts, resulting in the generation of harmful information. Through a series of experiments, we first validated the existence of the desire for continuity in LLMs, and further devised a straightforward yet powerful technique, such as incomplete sentences, negative priming, and cognitive dissonance scenarios, to demonstrate that even advanced LLMs struggle to prevent the generation of harmful information. In summary, our study uncovers the root of LLMs' vulnerabilities to adversarial attacks, hereby questioning the efficacy of solely relying on sophisticated alignment methods, and further advocates for a new training idea that integrates modal concepts alongside traditional amodal concepts, aiming to endow LLMs with a more nuanced understanding of real-world contexts and ethical considerations.



## **21. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

eess.IV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.09253v1) [paper-pdf](http://arxiv.org/pdf/2311.09253v1)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.



## **22. Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction**

cs.LG

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.05808v2) [paper-pdf](http://arxiv.org/pdf/2311.05808v2)

**Authors**: Shanghao Shi, Ning Wang, Yang Xiao, Chaoyu Zhang, Yi Shi, Y. Thomas Hou, Wenjing Lou

**Abstract**: Federated learning is known for its capability to safeguard participants' data privacy. However, recently emerged model inversion attacks (MIAs) have shown that a malicious parameter server can reconstruct individual users' local data samples through model updates. The state-of-the-art attacks either rely on computation-intensive search-based optimization processes to recover each input batch, making scaling difficult, or they involve the malicious parameter server adding extra modules before the global model architecture, rendering the attacks too conspicuous and easily detectable.   To overcome these limitations, we propose Scale-MIA, a novel MIA capable of efficiently and accurately recovering training samples of clients from the aggregated updates, even when the system is under the protection of a robust secure aggregation protocol. Unlike existing approaches treating models as black boxes, Scale-MIA recognizes the importance of the intricate architecture and inner workings of machine learning models. It identifies the latent space as the critical layer for breaching privacy and decomposes the complex recovery task into an innovative two-step process to reduce computation complexity. The first step involves reconstructing the latent space representations (LSRs) from the aggregated model updates using a closed-form inversion mechanism, leveraging specially crafted adversarial linear layers. In the second step, the whole input batches are recovered from the LSRs by feeding them into a fine-tuned generative decoder.   We implemented Scale-MIA on multiple commonly used machine learning models and conducted comprehensive experiments across various settings. The results demonstrate that Scale-MIA achieves excellent recovery performance on different datasets, exhibiting high reconstruction rates, accuracy, and attack efficiency on a larger scale compared to state-of-the-art MIAs.



## **23. Laccolith: Hypervisor-Based Adversary Emulation with Anti-Detection**

cs.CR

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08274v1) [paper-pdf](http://arxiv.org/pdf/2311.08274v1)

**Authors**: Vittorio Orbinato, Marco Carlo Feliciano, Domenico Cotroneo, Roberto Natella

**Abstract**: Advanced Persistent Threats (APTs) represent the most threatening form of attack nowadays since they can stay undetected for a long time. Adversary emulation is a proactive approach for preparing against these attacks. However, adversary emulation tools lack the anti-detection abilities of APTs. We introduce Laccolith, a hypervisor-based solution for adversary emulation with anti-detection to fill this gap. We also present an experimental study to compare Laccolith with MITRE CALDERA, a state-of-the-art solution for adversary emulation, against five popular anti-virus products. We found that CALDERA cannot evade detection, limiting the realism of emulated attacks, even when combined with a state-of-the-art anti-detection framework. Our experiments show that Laccolith can hide its activities from all the tested anti-virus products, thus making it suitable for realistic emulations.



## **24. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08268v1) [paper-pdf](http://arxiv.org/pdf/2311.08268v1)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on another white-box model, compromising generalization or jailbreak efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we offer detailed analysis and discussion from the perspective of prompt execution priority on the failure of LLMs' defense. We hope that our research can catalyze both the academic community and LLMs vendors towards the provision of safer and more regulated Large Language Models.



## **25. On The Relationship Between Universal Adversarial Attacks And Sparse Representations**

cs.CV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.08265v1) [paper-pdf](http://arxiv.org/pdf/2311.08265v1)

**Authors**: Dana Weitzner, Raja Giryes

**Abstract**: The prominent success of neural networks, mainly in computer vision tasks, is increasingly shadowed by their sensitivity to small, barely perceivable adversarial perturbations in image input.   In this work, we aim at explaining this vulnerability through the framework of sparsity.   We show the connection between adversarial attacks and sparse representations, with a focus on explaining the universality and transferability of adversarial examples in neural networks.   To this end, we show that sparse coding algorithms, and the neural network-based learned iterative shrinkage thresholding algorithm (LISTA) among them, suffer from this sensitivity, and that common attacks on neural networks can be expressed as attacks on the sparse representation of the input image. The phenomenon that we observe holds true also when the network is agnostic to the sparse representation and dictionary, and thus can provide a possible explanation for the universality and transferability of adversarial attacks.   The code is available at https://github.com/danawr/adversarial_attacks_and_sparse_representations.



## **26. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

cs.CR

Submitted to ICC 2023 conference

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.07946v1) [paper-pdf](http://arxiv.org/pdf/2311.07946v1)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between 9% and 66.5% for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.



## **27. Towards Improving Robustness Against Common Corruptions in Object Detectors Using Adversarial Contrastive Learning**

cs.CV

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.07928v1) [paper-pdf](http://arxiv.org/pdf/2311.07928v1)

**Authors**: Shashank Kotyan, Danilo Vasconcellos Vargas

**Abstract**: Neural networks have revolutionized various domains, exhibiting remarkable accuracy in tasks like natural language processing and computer vision. However, their vulnerability to slight alterations in input samples poses challenges, particularly in safety-critical applications like autonomous driving. Current approaches, such as introducing distortions during training, fall short in addressing unforeseen corruptions. This paper proposes an innovative adversarial contrastive learning framework to enhance neural network robustness simultaneously against adversarial attacks and common corruptions. By generating instance-wise adversarial examples and optimizing contrastive loss, our method fosters representations that resist adversarial perturbations and remain robust in real-world scenarios. Subsequent contrastive learning then strengthens the similarity between clean samples and their adversarial counterparts, fostering representations resistant to both adversarial attacks and common distortions. By focusing on improving performance under adversarial and real-world conditions, our approach aims to bolster the robustness of neural networks in safety-critical applications, such as autonomous vehicles navigating unpredictable weather conditions. We anticipate that this framework will contribute to advancing the reliability of neural networks in challenging environments, facilitating their widespread adoption in mission-critical scenarios.



## **28. Cooperative AI via Decentralized Commitment Devices**

cs.AI

NeurIPS 2023- Multi-Agent Security Workshop

**SubmitDate**: 2023-11-14    [abs](http://arxiv.org/abs/2311.07815v1) [paper-pdf](http://arxiv.org/pdf/2311.07815v1)

**Authors**: Xinyuan Sun, Davide Crapis, Matt Stephenson, Barnabé Monnot, Thomas Thiery, Jonathan Passerat-Palmbach

**Abstract**: Credible commitment devices have been a popular approach for robust multi-agent coordination. However, existing commitment mechanisms face limitations like privacy, integrity, and susceptibility to mediator or user strategic behavior. It is unclear if the cooperative AI techniques we study are robust to real-world incentives and attack vectors. However, decentralized commitment devices that utilize cryptography have been deployed in the wild, and numerous studies have shown their ability to coordinate algorithmic agents facing adversarial opponents with significant economic incentives, currently in the order of several million to billions of dollars. In this paper, we use examples in the decentralization and, in particular, Maximal Extractable Value (MEV) (arXiv:1904.05234) literature to illustrate the potential security issues in cooperative AI. We call for expanded research into decentralized commitments to advance cooperative AI capabilities for secure coordination in open environments and empirical testing frameworks to evaluate multi-agent coordination ability given real-world commitment constraints.



## **29. Parrot-Trained Adversarial Examples: Pushing the Practicality of Black-Box Audio Attacks against Speaker Recognition Models**

cs.SD

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07780v1) [paper-pdf](http://arxiv.org/pdf/2311.07780v1)

**Authors**: Rui Duan, Zhe Qu, Leah Ding, Yao Liu, Zhuo Lu

**Abstract**: Audio adversarial examples (AEs) have posed significant security challenges to real-world speaker recognition systems. Most black-box attacks still require certain information from the speaker recognition model to be effective (e.g., keeping probing and requiring the knowledge of similarity scores). This work aims to push the practicality of the black-box attacks by minimizing the attacker's knowledge about a target speaker recognition model. Although it is not feasible for an attacker to succeed with completely zero knowledge, we assume that the attacker only knows a short (or a few seconds) speech sample of a target speaker. Without any probing to gain further knowledge about the target model, we propose a new mechanism, called parrot training, to generate AEs against the target model. Motivated by recent advancements in voice conversion (VC), we propose to use the one short sentence knowledge to generate more synthetic speech samples that sound like the target speaker, called parrot speech. Then, we use these parrot speech samples to train a parrot-trained(PT) surrogate model for the attacker. Under a joint transferability and perception framework, we investigate different ways to generate AEs on the PT model (called PT-AEs) to ensure the PT-AEs can be generated with high transferability to a black-box target model with good human perceptual quality. Real-world experiments show that the resultant PT-AEs achieve the attack success rates of 45.8% - 80.8% against the open-source models in the digital-line scenario and 47.9% - 58.3% against smart devices, including Apple HomePod (Siri), Amazon Echo, and Google Home, in the over-the-air scenario.



## **30. Towards a robust and reliable deep learning approach for detection of compact binary mergers in gravitational wave data**

gr-qc

22 pages, 22 figures

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2306.11797v2) [paper-pdf](http://arxiv.org/pdf/2306.11797v2)

**Authors**: Shreejit Jadhav, Mihir Shrivastava, Sanjit Mitra

**Abstract**: The ability of deep learning (DL) approaches to learn generalised signal and noise models, coupled with their fast inference on GPUs, holds great promise for enhancing gravitational-wave (GW) searches in terms of speed, parameter space coverage, and search sensitivity. However, the opaque nature of DL models severely harms their reliability. In this work, we meticulously develop a DL model stage-wise and work towards improving its robustness and reliability. First, we address the problems in maintaining the purity of training data by deriving a new metric that better reflects the visual strength of the 'chirp' signal features in the data. Using a reduced, smooth representation obtained through a variational auto-encoder (VAE), we build a classifier to search for compact binary coalescence (CBC) signals. Our tests on real LIGO data show an impressive performance of the model. However, upon probing the robustness of the model through adversarial attacks, its simple failure modes were identified, underlining how such models can still be highly fragile. As a first step towards bringing robustness, we retrain the model in a novel framework involving a generative adversarial network (GAN). Over the course of training, the model learns to eliminate the primary modes of failure identified by the adversaries. Although absolute robustness is practically impossible to achieve, we demonstrate some fundamental improvements earned through such training, like sparseness and reduced degeneracy in the extracted features at different layers inside the model. We show that these gains are achieved at practically zero loss in terms of model performance on real LIGO data before and after GAN training. Through a direct search on 8.8 days of LIGO data, we recover two significant CBC events from GWTC-2.1, GW190519_153544 and GW190521_074359. We also report the search sensitivity obtained from an injection study.



## **31. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming**

cs.CL

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07689v1) [paper-pdf](http://arxiv.org/pdf/2311.07689v1)

**Authors**: Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao

**Abstract**: Red-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses. While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them. In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM. Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.



## **32. An Extensive Study on Adversarial Attack against Pre-trained Models of Code**

cs.CR

Accepted to ESEC/FSE 2023

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07553v1) [paper-pdf](http://arxiv.org/pdf/2311.07553v1)

**Authors**: Xiaohu Du, Ming Wen, Zichao Wei, Shangwen Wang, Hai Jin

**Abstract**: Transformer-based pre-trained models of code (PTMC) have been widely utilized and have achieved state-of-the-art performance in many mission-critical applications. However, they can be vulnerable to adversarial attacks through identifier substitution or coding style transformation, which can significantly degrade accuracy and may further incur security concerns. Although several approaches have been proposed to generate adversarial examples for PTMC, the effectiveness and efficiency of such approaches, especially on different code intelligence tasks, has not been well understood. To bridge this gap, this study systematically analyzes five state-of-the-art adversarial attack approaches from three perspectives: effectiveness, efficiency, and the quality of generated examples. The results show that none of the five approaches balances all these perspectives. Particularly, approaches with a high attack success rate tend to be time-consuming; the adversarial code they generate often lack naturalness, and vice versa. To address this limitation, we explore the impact of perturbing identifiers under different contexts and find that identifier substitution within for and if statements is the most effective. Based on these findings, we propose a new approach that prioritizes different types of statements for various tasks and further utilizes beam search to generate adversarial examples. Evaluation results show that it outperforms the state-of-the-art ALERT in terms of both effectiveness and efficiency while preserving the naturalness of the generated adversarial examples.



## **33. On the Robustness of Neural Collapse and the Neural Collapse of Robustness**

cs.LG

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07444v1) [paper-pdf](http://arxiv.org/pdf/2311.07444v1)

**Authors**: Jingtong Su, Ya Shi Zhang, Nikolaos Tsilivis, Julia Kempe

**Abstract**: Neural Collapse refers to the curious phenomenon in the end of training of a neural network, where feature vectors and classification weights converge to a very simple geometrical arrangement (a simplex). While it has been observed empirically in various cases and has been theoretically motivated, its connection with crucial properties of neural networks, like their generalization and robustness, remains unclear. In this work, we study the stability properties of these simplices. We find that the simplex structure disappears under small adversarial attacks, and that perturbed examples "leap" between simplex vertices. We further analyze the geometry of networks that are optimized to be robust against adversarial perturbations of the input, and find that Neural Collapse is a pervasive phenomenon in these cases as well, with clean and perturbed representations forming aligned simplices, and giving rise to a robust simple nearest-neighbor classifier. By studying the propagation of the amount of collapse inside the network, we identify novel properties of both robust and non-robust machine learning models, and show that earlier, unlike later layers maintain reliable simplices on perturbed data.



## **34. Transpose Attack: Stealing Datasets with Bidirectional Training**

cs.LG

NDSS24 paper

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07389v1) [paper-pdf](http://arxiv.org/pdf/2311.07389v1)

**Authors**: Guy Amit, Mosh Levy, Yisroel Mirsky

**Abstract**: Deep neural networks are normally executed in the forward direction. However, in this work, we identify a vulnerability that enables models to be trained in both directions and on different tasks. Adversaries can exploit this capability to hide rogue models within seemingly legitimate models. In addition, in this work we show that neural networks can be taught to systematically memorize and retrieve specific samples from datasets. Together, these findings expose a novel method in which adversaries can exfiltrate datasets from protected learning environments under the guise of legitimate models. We focus on the data exfiltration attack and show that modern architectures can be used to secretly exfiltrate tens of thousands of samples with high fidelity, high enough to compromise data privacy and even train new models. Moreover, to mitigate this threat we propose a novel approach for detecting infected models.



## **35. Untargeted Black-box Attacks for Social Recommendations**

cs.SI

Preprint. Under review

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07127v1) [paper-pdf](http://arxiv.org/pdf/2311.07127v1)

**Authors**: Wenqi Fan, Shijie Wang, Xiao-yong Wei, Xiaowei Mei, Qing Li

**Abstract**: The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on targeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework Multiattack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.



## **36. Adversarial Purification for Data-Driven Power System Event Classifiers with Diffusion Models**

eess.SY

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2311.07110v1) [paper-pdf](http://arxiv.org/pdf/2311.07110v1)

**Authors**: Yuanbin Cheng, Koji Yamashita, Jim Follum, Nanpeng Yu

**Abstract**: The global deployment of the phasor measurement units (PMUs) enables real-time monitoring of the power system, which has stimulated considerable research into machine learning-based models for event detection and classification. However, recent studies reveal that machine learning-based methods are vulnerable to adversarial attacks, which can fool the event classifiers by adding small perturbations to the raw PMU data. To mitigate the threats posed by adversarial attacks, research on defense strategies is urgently needed. This paper proposes an effective adversarial purification method based on the diffusion model to counter adversarial attacks on the machine learning-based power system event classifier. The proposed method includes two steps: injecting noise into the PMU data; and utilizing a pre-trained neural network to eliminate the added noise while simultaneously removing perturbations introduced by the adversarial attacks. The proposed adversarial purification method significantly increases the accuracy of the event classifier under adversarial attacks while satisfying the requirements of real-time operations. In addition, the theoretical analysis reveals that the proposed diffusion model-based adversarial purification method decreases the distance between the original and compromised PMU data, which reduces the impacts of adversarial attacks. The empirical results on a large-scale real-world PMU dataset validate the effectiveness and computational efficiency of the proposed adversarial purification method.



## **37. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

cs.CL

Under Review

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2310.14303v2) [paper-pdf](http://arxiv.org/pdf/2310.14303v2)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.



## **38. PATROL: Privacy-Oriented Pruning for Collaborative Inference Against Model Inversion Attacks**

cs.LG

**SubmitDate**: 2023-11-13    [abs](http://arxiv.org/abs/2307.10981v2) [paper-pdf](http://arxiv.org/pdf/2307.10981v2)

**Authors**: Shiwei Ding, Lan Zhang, Miao Pan, Xiaoyong Yuan

**Abstract**: Collaborative inference has been a promising solution to enable resource-constrained edge devices to perform inference using state-of-the-art deep neural networks (DNNs). In collaborative inference, the edge device first feeds the input to a partial DNN locally and then uploads the intermediate result to the cloud to complete the inference. However, recent research indicates model inversion attacks (MIAs) can reconstruct input data from intermediate results, posing serious privacy concerns for collaborative inference. Existing perturbation and cryptography techniques are inefficient and unreliable in defending against MIAs while performing accurate inference. This paper provides a viable solution, named PATROL, which develops privacy-oriented pruning to balance privacy, efficiency, and utility of collaborative inference. PATROL takes advantage of the fact that later layers in a DNN can extract more task-specific features. Given limited local resources for collaborative inference, PATROL intends to deploy more layers at the edge based on pruning techniques to enforce task-specific features for inference and reduce task-irrelevant but sensitive features for privacy preservation. To achieve privacy-oriented pruning, PATROL introduces two key components: Lipschitz regularization and adversarial reconstruction training, which increase the reconstruction errors by reducing the stability of MIAs and enhance the target inference model by adversarial training, respectively. On a real-world collaborative inference task, vehicle re-identification, we demonstrate the superior performance of PATROL in terms of against MIAs.



## **39. Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2311.06942v1) [paper-pdf](http://arxiv.org/pdf/2311.06942v1)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of contractive dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.



## **40. Facial Data Minimization: Shallow Model as Your Privacy Filter**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2310.15590v2) [paper-pdf](http://arxiv.org/pdf/2310.15590v2)

**Authors**: Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji

**Abstract**: Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.



## **41. Learning Globally Optimized Language Structure via Adversarial Training**

cs.CL

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2311.06771v1) [paper-pdf](http://arxiv.org/pdf/2311.06771v1)

**Authors**: Xuwang Yin

**Abstract**: Recent work has explored integrating autoregressive language models with energy-based models (EBMs) to enhance text generation capabilities. However, learning effective EBMs for text is challenged by the discrete nature of language. This work proposes an adversarial training strategy to address limitations in prior efforts. Specifically, an iterative adversarial attack algorithm is presented to generate negative samples for training the EBM by perturbing text from the autoregressive model. This aims to enable the EBM to suppress spurious modes outside the support of the data distribution. Experiments on an arithmetic sequence generation task demonstrate that the proposed adversarial training approach can substantially enhance the quality of generated sequences compared to prior methods. The results highlight the promise of adversarial techniques to improve discrete EBM training. Key contributions include: (1) an adversarial attack strategy tailored to text to generate negative samples, circumventing MCMC limitations; (2) an adversarial training algorithm for EBMs leveraging these attacks; (3) empirical validation of performance improvements on a sequence generation task.



## **42. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

cs.CV

26 pages, 17 figures, 1 table, accepted to Frontiers in Computer  Science, 2023

**SubmitDate**: 2023-11-12    [abs](http://arxiv.org/abs/2307.02881v5) [paper-pdf](http://arxiv.org/pdf/2307.02881v5)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating image probability density functions that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space-not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. We therefore consider popular generative models. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: the possibility to sample from this distribution with the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute its probability, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show how semantic interpretations are used to describe points on the manifold. To achieve this, we consider an emergent language framework that uses variational encoders for a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described as evolving semantic descriptions. We also show that such probabilistic descriptions (bounded) can be used to improve semantic consistency by constructing defences against adversarial attacks. We evaluate our methods with improved semantic robustness and OoD detection capability, explainable and editable semantic interpolation, and improved classification accuracy under patch attacks. We also discuss the limitation in diffusion models.



## **43. Privacy Risks Analysis and Mitigation in Federated Learning for Medical Images**

cs.LG

V1

**SubmitDate**: 2023-11-11    [abs](http://arxiv.org/abs/2311.06643v1) [paper-pdf](http://arxiv.org/pdf/2311.06643v1)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Federated learning (FL) is gaining increasing popularity in the medical domain for analyzing medical images, which is considered an effective technique to safeguard sensitive patient data and comply with privacy regulations. However, several recent studies have revealed that the default settings of FL may leak private training data under privacy attacks. Thus, it is still unclear whether and to what extent such privacy risks of FL exist in the medical domain, and if so, ``how to mitigate such risks?''. In this paper, first, we propose a holistic framework for Medical data Privacy risk analysis and mitigation in Federated Learning (MedPFL) to analyze privacy risks and develop effective mitigation strategies in FL for protecting private medical data. Second, we demonstrate the substantial privacy risks of using FL to process medical images, where adversaries can easily perform privacy attacks to reconstruct private medical images accurately. Third, we show that the defense approach of adding random noises may not always work effectively to protect medical images against privacy attacks in FL, which poses unique and pressing challenges associated with medical data for privacy protection.



## **44. Verifiable Learning for Robust Tree Ensembles**

cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023 with corrected typo in footnote 1

**SubmitDate**: 2023-11-11    [abs](http://arxiv.org/abs/2305.03626v4) [paper-pdf](http://arxiv.org/pdf/2305.03626v4)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.



## **45. Seeing is Believing: A Federated Learning Based Prototype to Detect Wireless Injection Attacks**

cs.IT

6 pages with 8 figures

**SubmitDate**: 2023-11-11    [abs](http://arxiv.org/abs/2311.06564v1) [paper-pdf](http://arxiv.org/pdf/2311.06564v1)

**Authors**: Aadil Hussain, Nitheesh Gundapu, Sarang Drugkar, Suraj Kiran, J. Harshan, Ranjitha Prasad

**Abstract**: Reactive injection attacks are a class of security threats in wireless networks wherein adversaries opportunistically inject spoofing packets in the frequency band of a client thereby forcing the base-station to deploy impersonation-detection methods. Towards circumventing such threats, we implement secret-key based physical-layer signalling methods at the clients which allow the base-stations to deploy machine learning (ML) models on their in-phase and quadrature samples at the baseband for attack detection. Using Adalm Pluto based software defined radios to implement the secret-key based signalling methods, we show that robust ML models can be designed at the base-stations. However, we also point out that, in practice, insufficient availability of training datasets at the base-stations can make these methods ineffective. Thus, we use a federated learning framework in the backhaul network, wherein a group of base-stations that need to protect their clients against reactive injection threats collaborate to refine their ML models by ensuring privacy on their datasets. Using a network of XBee devices to implement the backhaul network, experimental results on our federated learning setup shows significant enhancements in the detection accuracy, thus presenting wireless security as an excellent use-case for federated learning in 6G networks and beyond.



## **46. Flatness-aware Adversarial Attack**

cs.LG

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06423v1) [paper-pdf](http://arxiv.org/pdf/2311.06423v1)

**Authors**: Mingyuan Fan, Xiaodan Li, Cen Chen, Yinggui Wang

**Abstract**: The transferability of adversarial examples can be exploited to launch black-box attacks. However, adversarial examples often present poor transferability. To alleviate this issue, by observing that the diversity of inputs can boost transferability, input regularization based methods are proposed, which craft adversarial examples by combining several transformed inputs. We reveal that input regularization based methods make resultant adversarial examples biased towards flat extreme regions. Inspired by this, we propose an attack called flatness-aware adversarial attack (FAA) which explicitly adds a flatness-aware regularization term in the optimization target to promote the resultant adversarial examples towards flat extreme regions. The flatness-aware regularization term involves gradients of samples around the resultant adversarial examples but optimizing gradients requires the evaluation of Hessian matrix in high-dimension spaces which generally is intractable. To address the problem, we derive an approximate solution to circumvent the construction of Hessian matrix, thereby making FAA practical and cheap. Extensive experiments show the transferability of adversarial examples crafted by FAA can be considerably boosted compared with state-of-the-art baselines.



## **47. CALLOC: Curriculum Adversarial Learning for Secure and Robust Indoor Localization**

cs.LG

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06361v1) [paper-pdf](http://arxiv.org/pdf/2311.06361v1)

**Authors**: Danish Gufran, Sudeep Pasricha

**Abstract**: Indoor localization has become increasingly vital for many applications from tracking assets to delivering personalized services. Yet, achieving pinpoint accuracy remains a challenge due to variations across indoor environments and devices used to assist with localization. Another emerging challenge is adversarial attacks on indoor localization systems that not only threaten service integrity but also reduce localization accuracy. To combat these challenges, we introduce CALLOC, a novel framework designed to resist adversarial attacks and variations across indoor environments and devices that reduce system accuracy and reliability. CALLOC employs a novel adaptive curriculum learning approach with a domain specific lightweight scaled-dot product attention neural network, tailored for adversarial and variation resilience in practical use cases with resource constrained mobile devices. Experimental evaluations demonstrate that CALLOC can achieve improvements of up to 6.03x in mean error and 4.6x in worst-case error against state-of-the-art indoor localization frameworks, across diverse building floorplans, mobile devices, and adversarial attacks scenarios.



## **48. SneakyPrompt: Jailbreaking Text-to-image Generative Models**

cs.LG

To appear in the Proceedings of the IEEE Symposium on Security and  Privacy (Oakland), 2024

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2305.12082v3) [paper-pdf](http://arxiv.org/pdf/2305.12082v3)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E raise many ethical concerns due to the generation of harmful images such as Not-Safe-for-Work (NSFW) ones. To address these ethical concerns, safety filters are often adopted to prevent the generation of NSFW images. In this work, we propose SneakyPrompt, the first automated attack framework, to jailbreak text-to-image generative models such that they generate NSFW images even if safety filters are adopted. Given a prompt that is blocked by a safety filter, SneakyPrompt repeatedly queries the text-to-image generative model and strategically perturbs tokens in the prompt based on the query results to bypass the safety filter. Specifically, SneakyPrompt utilizes reinforcement learning to guide the perturbation of tokens. Our evaluation shows that SneakyPrompt successfully jailbreaks DALL$\cdot$E 2 with closed-box safety filters to generate NSFW images. Moreover, we also deploy several state-of-the-art, open-source safety filters on a Stable Diffusion model. Our evaluation shows that SneakyPrompt not only successfully generates NSFW images, but also outperforms existing text adversarial attacks when extended to jailbreak text-to-image generative models, in terms of both the number of queries and qualities of the generated NSFW images. SneakyPrompt is open-source and available at this repository: \url{https://github.com/Yuchen413/text2image_safety}.



## **49. Triad: Trusted Timestamps in Untrusted Environments**

cs.CR

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06156v1) [paper-pdf](http://arxiv.org/pdf/2311.06156v1)

**Authors**: Gabriel P. Fernandez, Andrey Brito, Christof Fetzer

**Abstract**: We aim to provide trusted time measurement mechanisms to applications and cloud infrastructure deployed in environments that could harbor potential adversaries, including the hardware infrastructure provider. Despite Trusted Execution Environments (TEEs) providing multiple security functionalities, timestamps from the Operating System are not covered. Nevertheless, some services require time for validating permissions or ordering events. To address that need, we introduce Triad, a trusted timestamp dispatcher of time readings. The solution provides trusted timestamps enforced by mutually supportive enclave-based clock servers that create a continuous trusted timeline. We leverage enclave properties such as forced exits and CPU-based counters to mitigate attacks on the server's timestamp counters. Triad produces trusted, confidential, monotonically-increasing timestamps with bounded error and desirable, non-trivial properties. Our implementation relies on Intel SGX and SCONE, allowing transparent usage. We evaluate Triad's error and behavior in multiple dimensions.



## **50. Fight Fire with Fire: Combating Adversarial Patch Attacks using Pattern-randomized Defensive Patches**

cs.CV

**SubmitDate**: 2023-11-10    [abs](http://arxiv.org/abs/2311.06122v1) [paper-pdf](http://arxiv.org/pdf/2311.06122v1)

**Authors**: Jianan Feng, Jiachun Li, Changqing Miao, Jianjun Huang, Wei You, Wenchang Shi, Bin Liang

**Abstract**: Object detection has found extensive applications in various tasks, but it is also susceptible to adversarial patch attacks. Existing defense methods often necessitate modifications to the target model or result in unacceptable time overhead. In this paper, we adopt a counterattack approach, following the principle of "fight fire with fire," and propose a novel and general methodology for defending adversarial attacks. We utilize an active defense strategy by injecting two types of defensive patches, canary and woodpecker, into the input to proactively probe or weaken potential adversarial patches without altering the target model. Moreover, inspired by randomization techniques employed in software security, we employ randomized canary and woodpecker injection patterns to defend against defense-aware attacks. The effectiveness and practicality of the proposed method are demonstrated through comprehensive experiments. The results illustrate that canary and woodpecker achieve high performance, even when confronted with unknown attack methods, while incurring limited time overhead. Furthermore, our method also exhibits sufficient robustness against defense-aware attacks, as evidenced by adaptive attack experiments.



