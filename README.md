# Latest Adversarial Attack Papers
**update at 2024-12-06 16:11:51**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation**

cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04415v1) [paper-pdf](http://arxiv.org/pdf/2412.04415v1)

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Yasuhiro Yoshida, Victor Bian

**Abstract**: AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.



## **2. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2406.12259v2) [paper-pdf](http://arxiv.org/pdf/2406.12259v2)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **3. Machine Theory of Mind for Autonomous Cyber-Defence**

cs.LG

29 pages, 17 figures, 12 tables

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04367v1) [paper-pdf](http://arxiv.org/pdf/2412.04367v1)

**Authors**: Luke Swaby, Matthew Stewart, Daniel Harrold, Chris Willis, Gregory Palmer

**Abstract**: Intelligent autonomous agents hold much potential for the domain of cyber-security. However, due to many state-of-the-art approaches relying on uninterpretable black-box models, there is growing demand for methods that offer stakeholders clear and actionable insights into their latent beliefs and motivations. To address this, we evaluate Theory of Mind (ToM) approaches for Autonomous Cyber Operations. Upon learning a robust prior, ToM models can predict an agent's goals, behaviours, and contextual beliefs given only a handful of past behaviour observations. In this paper, we introduce a novel Graph Neural Network (GNN)-based ToM architecture tailored for cyber-defence, Graph-In, Graph-Out (GIGO)-ToM, which can accurately predict both the targets and attack trajectories of adversarial cyber agents over arbitrary computer network topologies. To evaluate the latter, we propose a novel extension of the Wasserstein distance for measuring the similarity of graph-based probability distributions. Whereas the standard Wasserstein distance lacks a fixed reference scale, we introduce a graph-theoretic normalization factor that enables a standardized comparison between networks of different sizes. We furnish this metric, which we term the Network Transport Distance (NTD), with a weighting function that emphasizes predictions according to custom node features, allowing network operators to explore arbitrary strategic considerations. Benchmarked against a Graph-In, Dense-Out (GIDO)-ToM architecture in an abstract cyber-defence environment, our empirical evaluations show that GIGO-ToM can accurately predict the goals and behaviours of various unseen cyber-attacking agents across a range of network topologies, as well as learn embeddings that can effectively characterize their policies.



## **4. PBP: Post-training Backdoor Purification for Malware Classifiers**

cs.LG

Accepted at NDSS 2025

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03441v2) [paper-pdf](http://arxiv.org/pdf/2412.03441v2)

**Authors**: Dung Thuy Nguyen, Ngoc N. Tran, Taylor T. Johnson, Kevin Leach

**Abstract**: In recent years, the rise of machine learning (ML) in cybersecurity has brought new challenges, including the increasing threat of backdoor poisoning attacks on ML malware classifiers. For instance, adversaries could inject malicious samples into public malware repositories, contaminating the training data and potentially misclassifying malware by the ML model. Current countermeasures predominantly focus on detecting poisoned samples by leveraging disagreements within the outputs of a diverse set of ensemble models on training data points. However, these methods are not suitable for scenarios where Machine Learning-as-a-Service (MLaaS) is used or when users aim to remove backdoors from a model after it has been trained. Addressing this scenario, we introduce PBP, a post-training defense for malware classifiers that mitigates various types of backdoor embeddings without assuming any specific backdoor embedding mechanism. Our method exploits the influence of backdoor attacks on the activation distribution of neural networks, independent of the trigger-embedding method. In the presence of a backdoor attack, the activation distribution of each layer is distorted into a mixture of distributions. By regulating the statistics of the batch normalization layers, we can guide a backdoored model to perform similarly to a clean one. Our method demonstrates substantial advantages over several state-of-the-art methods, as evidenced by experiments on two datasets, two types of backdoor methods, and various attack configurations. Notably, our approach requires only a small portion of the training data -- only 1\% -- to purify the backdoor and reduce the attack success rate from 100\% to almost 0\%, a 100-fold improvement over the baseline methods. Our code is available at \url{https://github.com/judydnguyen/pbp-backdoor-purification-official}.



## **5. On the Lack of Robustness of Binary Function Similarity Systems**

cs.CR

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.04163v1) [paper-pdf](http://arxiv.org/pdf/2412.04163v1)

**Authors**: Gianluca Capozzi, Tong Tang, Jie Wan, Ziqi Yang, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Lorenzo Cavallaro, Leonardo Querzoni

**Abstract**: Binary function similarity, which often relies on learning-based algorithms to identify what functions in a pool are most similar to a given query function, is a sought-after topic in different communities, including machine learning, software engineering, and security. Its importance stems from the impact it has in facilitating several crucial tasks, from reverse engineering and malware analysis to automated vulnerability detection. Whereas recent work cast light around performance on this long-studied problem, the research landscape remains largely lackluster in understanding the resiliency of the state-of-the-art machine learning models against adversarial attacks. As security requires to reason about adversaries, in this work we assess the robustness of such models through a simple yet effective black-box greedy attack, which modifies the topology and the content of the control flow of the attacked functions. We demonstrate that this attack is successful in compromising all the models, achieving average attack success rates of 57.06% and 95.81% depending on the problem settings (targeted and untargeted attacks). Our findings are insightful: top performance on clean data does not necessarily relate to top robustness properties, which explicitly highlights performance-robustness trade-offs one should consider when deploying such models, calling for further research.



## **6. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

cs.LG

v2: Updated with changes from peer review rebuttal. v1: Version under  peer review

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.02785v2) [paper-pdf](http://arxiv.org/pdf/2411.02785v2)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt. Source code and data: https://github.com/uiuc-focal-lab/stochastic-monkeys/



## **7. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

eess.SP

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2411.18220v2) [paper-pdf](http://arxiv.org/pdf/2411.18220v2)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.



## **8. Safeguarding Text-to-Image Generation via Inference-Time Prompt-Noise Optimization**

cs.CV

**SubmitDate**: 2024-12-05    [abs](http://arxiv.org/abs/2412.03876v1) [paper-pdf](http://arxiv.org/pdf/2412.03876v1)

**Authors**: Jiangweizhi Peng, Zhiwei Tang, Gaowen Liu, Charles Fleming, Mingyi Hong

**Abstract**: Text-to-Image (T2I) diffusion models are widely recognized for their ability to generate high-quality and diverse images based on text prompts. However, despite recent advances, these models are still prone to generating unsafe images containing sensitive or inappropriate content, which can be harmful to users. Current efforts to prevent inappropriate image generation for diffusion models are easy to bypass and vulnerable to adversarial attacks. How to ensure that T2I models align with specific safety goals remains a significant challenge. In this work, we propose a novel, training-free approach, called Prompt-Noise Optimization (PNO), to mitigate unsafe image generation. Our method introduces a novel optimization framework that leverages both the continuous prompt embedding and the injected noise trajectory in the sampling process to generate safe images. Extensive numerical results demonstrate that our framework achieves state-of-the-art performance in suppressing toxic image generations and demonstrates robustness to adversarial attacks, without needing to tune the model parameters. Furthermore, compared with existing methods, PNO uses comparable generation time while offering the best tradeoff between the conflicting goals of safe generation and prompt-image alignment.



## **9. NODE-AdvGAN: Improving the transferability and perceptual similarity of adversarial examples by dynamic-system-driven adversarial generative model**

cs.LG

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03539v1) [paper-pdf](http://arxiv.org/pdf/2412.03539v1)

**Authors**: Xinheng Xie, Yue Wu, Cuiyu He

**Abstract**: Understanding adversarial examples is crucial for improving the model's robustness, as they introduce imperceptible perturbations that deceive models. Effective adversarial examples, therefore, offer the potential to train more robust models by removing their singularities. We propose NODE-AdvGAN, a novel approach that treats adversarial generation as a continuous process and employs a Neural Ordinary Differential Equation (NODE) for simulating the dynamics of the generator. By mimicking the iterative nature of traditional gradient-based methods, NODE-AdvGAN generates smoother and more precise perturbations that preserve high perceptual similarity when added to benign images. We also propose a new training strategy, NODE-AdvGAN-T, which enhances transferability in black-box attacks by effectively tuning noise parameters during training. Experiments demonstrate that NODE-AdvGAN and NODE-AdvGAN-T generate more effective adversarial examples that achieve higher attack success rates while preserving better perceptual quality than traditional GAN-based methods.



## **10. Pre-trained Multiple Latent Variable Generative Models are good defenders against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03453v1) [paper-pdf](http://arxiv.org/pdf/2412.03453v1)

**Authors**: Dario Serez, Marco Cristani, Alessio Del Bue, Vittorio Murino, Pietro Morerio

**Abstract**: Attackers can deliberately perturb classifiers' input with subtle noise, altering final predictions. Among proposed countermeasures, adversarial purification employs generative networks to preprocess input images, filtering out adversarial noise. In this study, we propose specific generators, defined Multiple Latent Variable Generative Models (MLVGMs), for adversarial purification. These models possess multiple latent variables that naturally disentangle coarse from fine features. Taking advantage of these properties, we autoencode images to maintain class-relevant information, while discarding and re-sampling any detail, including adversarial noise. The procedure is completely training-free, exploring the generalization abilities of pre-trained MLVGMs on the adversarial purification downstream task. Despite the lack of large models, trained on billions of samples, we show that smaller MLVGMs are already competitive with traditional methods, and can be used as foundation models. Official code released at https://github.com/SerezD/gen_adversarial.



## **11. State Frequency Estimation for Anomaly Detection**

cs.LG

9 pages

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03442v1) [paper-pdf](http://arxiv.org/pdf/2412.03442v1)

**Authors**: Clinton Cao, Agathe Blaise, Annibale Panichella, Sicco Verwer

**Abstract**: Many works have studied the efficacy of state machines for detecting anomalies within NetFlows. These works typically learn a model from unlabeled data and compute anomaly scores for arbitrary traces based on their likelihood of occurrence or how well they fit within the model. However, these methods do not dynamically adapt their scores based on the traces seen at test time. This becomes a problem when an adversary produces seemingly common traces in their attack, causing the model to miss the detection by assigning low anomaly scores. We propose SEQUENT, a new approach that uses the state visit frequency to adapt its scoring for anomaly detection dynamically. SEQUENT subsequently uses the scores to generate root causes for anomalies. These allow the grouping of alarms and simplify the analysis of anomalies. Our evaluation of SEQUENT on three NetFlow datasets indicates that our approach outperforms existing methods, demonstrating its effectiveness in detecting anomalies.



## **12. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

cs.CL

Accepted at the Safe Generative AI Workshop @ NeurIPS 2024

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03235v1) [paper-pdf](http://arxiv.org/pdf/2412.03235v1)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.



## **13. Testing Neural Network Verifiers: A Soundness Benchmark with Hidden Counterexamples**

cs.LG

Preprint

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03154v1) [paper-pdf](http://arxiv.org/pdf/2412.03154v1)

**Authors**: Xingjian Zhou, Hongji Xu, Andy Xu, Zhouxing Shi, Cho-Jui Hsieh, Huan Zhang

**Abstract**: In recent years, many neural network (NN) verifiers have been developed to formally verify certain properties of neural networks such as robustness. Although many benchmarks have been constructed to evaluate the performance of NN verifiers, they typically lack a ground-truth for hard instances where no current verifier can verify and no counterexample can be found, which makes it difficult to check the soundness of a new verifier if it claims to verify hard instances which no other verifier can do. We propose to develop a soundness benchmark for NN verification. Our benchmark contains instances with deliberately inserted counterexamples while we also try to hide the counterexamples from regular adversarial attacks which can be used for finding counterexamples. We design a training method to produce neural networks with such hidden counterexamples. Our benchmark aims to be used for testing the soundness of NN verifiers and identifying falsely claimed verifiability when it is known that hidden counterexamples exist. We systematically construct our benchmark and generate instances across diverse model architectures, activation functions, input sizes, and perturbation radii. We demonstrate that our benchmark successfully identifies bugs in state-of-the-art NN verifiers, as well as synthetic bugs, providing a crucial step toward enhancing the reliability of testing NN verifiers. Our code is available at https://github.com/MVP-Harry/SoundnessBench and our benchmark is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.



## **14. Pay Attention to the Robustness of Chinese Minority Language Models! Syllable-level Textual Adversarial Attack on Tibetan Script**

cs.CL

Revised Version; Accepted at ACL 2023 Workshop on TrustNLP

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.02323v2) [paper-pdf](http://arxiv.org/pdf/2412.02323v2)

**Authors**: Xi Cao, Dolma Dawa, Nuo Qun, Trashi Nyima

**Abstract**: The textual adversarial attack refers to an attack method in which the attacker adds imperceptible perturbations to the original texts by elaborate design so that the NLP (natural language processing) model produces false judgments. This method is also used to evaluate the robustness of NLP models. Currently, most of the research in this field focuses on English, and there is also a certain amount of research on Chinese. However, to the best of our knowledge, there is little research targeting Chinese minority languages. Textual adversarial attacks are a new challenge for the information processing of Chinese minority languages. In response to this situation, we propose a Tibetan syllable-level black-box textual adversarial attack called TSAttacker based on syllable cosine distance and scoring mechanism. And then, we conduct TSAttacker on six models generated by fine-tuning two PLMs (pre-trained language models) for three downstream tasks. The experiment results show that TSAttacker is effective and generates high-quality adversarial samples. In addition, the robustness of the involved models still has much room for improvement.



## **15. Less is More: A Stealthy and Efficient Adversarial Attack Method for DRL-based Autonomous Driving Policies**

cs.LG

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2412.03051v1) [paper-pdf](http://arxiv.org/pdf/2412.03051v1)

**Authors**: Junchao Fan, Xuyang Lei, Xiaolin Chang, Jelena Mišić, Vojislav B. Mišić

**Abstract**: Despite significant advancements in deep reinforcement learning (DRL)-based autonomous driving policies, these policies still exhibit vulnerability to adversarial attacks. This vulnerability poses a formidable challenge to the practical deployment of these policies in autonomous driving. Designing effective adversarial attacks is an indispensable prerequisite for enhancing the robustness of these policies. In view of this, we present a novel stealthy and efficient adversarial attack method for DRL-based autonomous driving policies. Specifically, we introduce a DRL-based adversary designed to trigger safety violations (e.g., collisions) by injecting adversarial samples at critical moments. We model the attack as a mixed-integer optimization problem and formulate it as a Markov decision process. Then, we train the adversary to learn the optimal policy for attacking at critical moments without domain knowledge. Furthermore, we introduce attack-related information and a trajectory clipping method to enhance the learning capability of the adversary. Finally, we validate our method in an unprotected left-turn scenario across different traffic densities. The experimental results show that our method achieves more than 90% collision rate within three attacks in most cases. Furthermore, our method achieves more than 130% improvement in attack efficiency compared to the unlimited attack method.



## **16. AED-PADA:Improving Generalizability of Adversarial Example Detection via Principal Adversarial Domain Adaptation**

cs.CV

**SubmitDate**: 2024-12-04    [abs](http://arxiv.org/abs/2404.12635v2) [paper-pdf](http://arxiv.org/pdf/2404.12635v2)

**Authors**: Heqi Peng, Yunhong Wang, Ruijie Yang, Beichen Li, Rui Wang, Yuanfang Guo

**Abstract**: Adversarial example detection, which can be conveniently applied in many scenarios, is important in the area of adversarial defense. Unfortunately, existing detection methods suffer from poor generalization performance, because their training process usually relies on the examples generated from a single known adversarial attack and there exists a large discrepancy between the training and unseen testing adversarial examples. To address this issue, we propose a novel method, named Adversarial Example Detection via Principal Adversarial Domain Adaptation (AED-PADA). Specifically, our approach identifies the Principal Adversarial Domains (PADs), i.e., a combination of features of the adversarial examples generated by different attacks, which possesses a large portion of the entire adversarial feature space. Subsequently, we pioneer to exploit Multi-source Unsupervised Domain Adaptation in adversarial example detection, with PADs as the source domains. Experimental results demonstrate the superior generalization ability of our proposed AED-PADA. Note that this superiority is particularly achieved in challenging scenarios characterized by employing the minimal magnitude constraint for the perturbations.



## **17. Exploiting the Uncoordinated Privacy Protections of Eye Tracking and VR Motion Data for Unauthorized User Identification**

cs.HC

11 pages, 3 figures

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2411.12766v2) [paper-pdf](http://arxiv.org/pdf/2411.12766v2)

**Authors**: Samantha Aziz, Oleg Komogortsev

**Abstract**: Virtual reality (VR) devices use a variety of sensors to capture a rich body of user-generated data. This data can be misused by malicious parties to covertly infer information about the user. Privacy-enhancing techniques that seek to reduce the amount of personally identifying information in sensor data are typically developed for a subset of data streams that are available on the platform, without consideration for the auxiliary information that may be readily available from other sensors. In this paper, we evaluate whether body motion data can be used to circumvent the privacy protections applied to eye tracking data to enable user identification on a VR platform, and vice versa. We empirically show that eye tracking, headset tracking, and hand tracking data are not only informative for inferring user identity on their own, but contain complementary information that can increase the rate of successful user identification. Most importantly, we demonstrate that applying privacy protections to only a subset of the data available in VR can create an opportunity for an adversary to bypass those privacy protections by using other unprotected data streams that are available on the platform, performing a user identification attack as accurately as though a privacy mechanism was never applied. These results highlight a new privacy consideration at the intersection between eye tracking and VR, and emphasizes the need for privacy-enhancing techniques that address multiple technologies comprehensively.



## **18. Gaussian Splatting Under Attack: Investigating Adversarial Noise in 3D Objects**

cs.CV

Accepted to Safe Generative AI Workshop @ NeurIPS 2024:  https://neurips.cc/virtual/2024/workshop/84705

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02803v1) [paper-pdf](http://arxiv.org/pdf/2412.02803v1)

**Authors**: Abdurrahman Zeybey, Mehmet Ergezer, Tommy Nguyen

**Abstract**: 3D Gaussian Splatting has advanced radiance field reconstruction, enabling high-quality view synthesis and fast rendering in 3D modeling. While adversarial attacks on object detection models are well-studied for 2D images, their impact on 3D models remains underexplored. This work introduces the Masked Iterative Fast Gradient Sign Method (M-IFGSM), designed to generate adversarial noise targeting the CLIP vision-language model. M-IFGSM specifically alters the object of interest by focusing perturbations on masked regions, degrading the performance of CLIP's zero-shot object detection capability when applied to 3D models. Using eight objects from the Common Objects 3D (CO3D) dataset, we demonstrate that our method effectively reduces the accuracy and confidence of the model, with adversarial noise being nearly imperceptible to human observers. The top-1 accuracy in original model renders drops from 95.4\% to 12.5\% for train images and from 91.2\% to 35.4\% for test images, with confidence levels reflecting this shift from true classification to misclassification, underscoring the risks of adversarial attacks on 3D models in applications such as autonomous driving, robotics, and surveillance. The significance of this research lies in its potential to expose vulnerabilities in modern 3D vision models, including radiance fields, prompting the development of more robust defenses and security measures in critical real-world applications.



## **19. Hijacking Vision-and-Language Navigation Agents with Adversarial Environmental Attacks**

cs.CV

Accepted by WACV 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02795v1) [paper-pdf](http://arxiv.org/pdf/2412.02795v1)

**Authors**: Zijiao Yang, Xiangxi Shi, Eric Slyman, Stefan Lee

**Abstract**: Assistive embodied agents that can be instructed in natural language to perform tasks in open-world environments have the potential to significantly impact labor tasks like manufacturing or in-home care -- benefiting the lives of those who come to depend on them. In this work, we consider how this benefit might be hijacked by local modifications in the appearance of the agent's operating environment. Specifically, we take the popular Vision-and-Language Navigation (VLN) task as a representative setting and develop a whitebox adversarial attack that optimizes a 3D attack object's appearance to induce desired behaviors in pretrained VLN agents that observe it in the environment. We demonstrate that the proposed attack can cause VLN agents to ignore their instructions and execute alternative actions after encountering the attack object -- even for instructions and agent paths not considered when optimizing the attack. For these novel settings, we find our attacks can induce early-termination behaviors or divert an agent along an attacker-defined multi-step trajectory. Under both conditions, environmental attacks significantly reduce agent capabilities to successfully follow user instructions.



## **20. Defending Against Diverse Attacks in Federated Learning Through Consensus-Based Bi-Level Optimization**

cs.LG

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02535v1) [paper-pdf](http://arxiv.org/pdf/2412.02535v1)

**Authors**: Nicolás García Trillos, Aditya Kumar Akash, Sixu Li, Konstantin Riedl, Yuhua Zhu

**Abstract**: Adversarial attacks pose significant challenges in many machine learning applications, particularly in the setting of distributed training and federated learning, where malicious agents seek to corrupt the training process with the goal of jeopardizing and compromising the performance and reliability of the final models. In this paper, we address the problem of robust federated learning in the presence of such attacks by formulating the training task as a bi-level optimization problem. We conduct a theoretical analysis of the resilience of consensus-based bi-level optimization (CB$^2$O), an interacting multi-particle metaheuristic optimization method, in adversarial settings. Specifically, we provide a global convergence analysis of CB$^2$O in mean-field law in the presence of malicious agents, demonstrating the robustness of CB$^2$O against a diverse range of attacks. Thereby, we offer insights into how specific hyperparameter choices enable to mitigate adversarial effects. On the practical side, we extend CB$^2$O to the clustered federated learning setting by proposing FedCB$^2$O, a novel interacting multi-particle system, and design a practical algorithm that addresses the demands of real-world applications. Extensive experiments demonstrate the robustness of the FedCB$^2$O algorithm against label-flipping attacks in decentralized clustered federated learning scenarios, showcasing its effectiveness in practical contexts.



## **21. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

cs.CL

Review Version; Submitted to ICASSP 2025

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02371v1) [paper-pdf](http://arxiv.org/pdf/2412.02371v1)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.



## **22. Multi-Granularity Tibetan Textual Adversarial Attack Method Based on Masked Language Model**

cs.CL

Revised Version; Accepted at WWW 2024 Workshop on SocialNLP

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02343v1) [paper-pdf](http://arxiv.org/pdf/2412.02343v1)

**Authors**: Xi Cao, Nuo Qun, Quzong Gesang, Yulei Zhu, Trashi Nyima

**Abstract**: In social media, neural network models have been applied to hate speech detection, sentiment analysis, etc., but neural network models are susceptible to adversarial attacks. For instance, in a text classification task, the attacker elaborately introduces perturbations to the original texts that hardly alter the original semantics in order to trick the model into making different predictions. By studying textual adversarial attack methods, the robustness of language models can be evaluated and then improved. Currently, most of the research in this field focuses on English, and there is also a certain amount of research on Chinese. However, there is little research targeting Chinese minority languages. With the rapid development of artificial intelligence technology and the emergence of Chinese minority language models, textual adversarial attacks become a new challenge for the information processing of Chinese minority languages. In response to this situation, we propose a multi-granularity Tibetan textual adversarial attack method based on masked language models called TSTricker. We utilize the masked language models to generate candidate substitution syllables or words, adopt the scoring mechanism to determine the substitution order, and then conduct the attack method on several fine-tuned victim models. The experimental results show that TSTricker reduces the accuracy of the classification models by more than 28.70% and makes the classification models change the predictions of more than 90.60% of the samples, which has an evidently higher attack effect than the baseline method.



## **23. Sustainable Self-evolution Adversarial Training**

cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02270v1) [paper-pdf](http://arxiv.org/pdf/2412.02270v1)

**Authors**: Wenxuan Wang, Chenglei Wang, Huihui Qi, Menghao Ye, Xuelin Qian, Peng Wang, Yanning Zhang

**Abstract**: With the wide application of deep neural network models in various computer vision tasks, there has been a proliferation of adversarial example generation strategies aimed at deeply exploring model security. However, existing adversarial training defense models, which rely on single or limited types of attacks under a one-time learning process, struggle to adapt to the dynamic and evolving nature of attack methods. Therefore, to achieve defense performance improvements for models in long-term applications, we propose a novel Sustainable Self-Evolution Adversarial Training (SSEAT) framework. Specifically, we introduce a continual adversarial defense pipeline to realize learning from various kinds of adversarial examples across multiple stages. Additionally, to address the issue of model catastrophic forgetting caused by continual learning from ongoing novel attacks, we propose an adversarial data replay module to better select more diverse and key relearning data. Furthermore, we design a consistency regularization strategy to encourage current defense models to learn more from previously trained ones, guiding them to retain more past knowledge and maintain accuracy on clean samples. Extensive experiments have been conducted to verify the efficacy of the proposed SSEAT defense method, which demonstrates superior defense performance and classification accuracy compared to competitors.



## **24. Guardian of the Ensembles: Introducing Pairwise Adversarially Robust Loss for Resisting Adversarial Attacks in DNN Ensembles**

cs.LG

Accepted at IEEE/CVF Winter Conference on Applications of Computer  Vision (WACV 2025)

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2112.04948v2) [paper-pdf](http://arxiv.org/pdf/2112.04948v2)

**Authors**: Shubhi Shukla, Subhadeep Dalui, Manaar Alam, Shubhajit Datta, Arijit Mondal, Debdeep Mukhopadhyay, Partha Pratim Chakrabarti

**Abstract**: Adversarial attacks rely on transferability, where an adversarial example (AE) crafted on a surrogate classifier tends to mislead a target classifier. Recent ensemble methods demonstrate that AEs are less likely to mislead multiple classifiers in an ensemble. This paper proposes a new ensemble training using a Pairwise Adversarially Robust Loss (PARL) that by construction produces an ensemble of classifiers with diverse decision boundaries. PARL utilizes outputs and gradients of each layer with respect to network parameters in every classifier within the ensemble simultaneously. PARL is demonstrated to achieve higher robustness against black-box transfer attacks than previous ensemble methods as well as adversarial training without adversely affecting clean example accuracy. Extensive experiments using standard Resnet20, WideResnet28-10 classifiers demonstrate the robustness of PARL against state-of-the-art adversarial attacks. While maintaining similar clean accuracy and lesser training time, the proposed architecture has a 24.8% increase in robust accuracy ($\epsilon$ = 0.07) from the state-of-the art method.



## **25. Privacy-Preserving Federated Learning via Homomorphic Adversarial Networks**

cs.CR

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.01650v2) [paper-pdf](http://arxiv.org/pdf/2412.01650v2)

**Authors**: Wenhan Dong, Chao Lin, Xinlei He, Xinyi Huang, Shengmin Xu

**Abstract**: Privacy-preserving federated learning (PPFL) aims to train a global model for multiple clients while maintaining their data privacy. However, current PPFL protocols exhibit one or more of the following insufficiencies: considerable degradation in accuracy, the requirement for sharing keys, and cooperation during the key generation or decryption processes. As a mitigation, we develop the first protocol that utilizes neural networks to implement PPFL, as well as incorporating an Aggregatable Hybrid Encryption scheme tailored to the needs of PPFL. We name these networks as Homomorphic Adversarial Networks (HANs) which demonstrate that neural networks are capable of performing tasks similar to multi-key homomorphic encryption (MK-HE) while solving the problems of key distribution and collaborative decryption. Our experiments show that HANs are robust against privacy attacks. Compared with non-private federated learning, experiments conducted on multiple datasets demonstrate that HANs exhibit a negligible accuracy loss (at most 1.35%). Compared to traditional MK-HE schemes, HANs increase encryption aggregation speed by 6,075 times while incurring a 29.2 times increase in communication overhead.



## **26. Investigating Privacy Leakage in Dimensionality Reduction Methods via Reconstruction Attack**

cs.CR

Major revision

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2408.17151v2) [paper-pdf](http://arxiv.org/pdf/2408.17151v2)

**Authors**: Chayadon Lumbut, Donlapark Ponnoprat

**Abstract**: This study investigates privacy leakage in dimensionality reduction methods through a novel machine learning-based reconstruction attack. Employing an informed adversary threat model, we develop a neural network capable of reconstructing high-dimensional data from low-dimensional embeddings.   We evaluate six popular dimensionality reduction techniques: PCA, sparse random projection (SRP), multidimensional scaling (MDS), Isomap, t-SNE, and UMAP. Using both MNIST and NIH Chest X-ray datasets, we perform a qualitative analysis to identify key factors affecting reconstruction quality. Furthermore, we assess the effectiveness of an additive noise mechanism in mitigating these reconstruction attacks. Our experimental results on both datasets reveal that the attack is effective against deterministic methods (PCA and Isomap), but ineffective against methods that employ random initialization (SRP, MDS, t-SNE and UMAP). When adding the images with large noises before performing PCA or Isomap, the attack produced severely distorted reconstructions. In contrast, for the other four methods, the reconstructions still show some recognizable features, though they bear little resemblance to the original images.



## **27. Underload: Defending against Latency Attacks for Object Detectors on Edge Devices**

cs.CV

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02171v1) [paper-pdf](http://arxiv.org/pdf/2412.02171v1)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.



## **28. Compromising the Intelligence of Modern DNNs: On the Effectiveness of Targeted RowPress**

cs.AR

8 Pages, 7 Figures, 1 Table

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2412.02156v1) [paper-pdf](http://arxiv.org/pdf/2412.02156v1)

**Authors**: Ranyang Zhou, Jacqueline T. Liu, Sabbir Ahmed, Shaahin Angizi, Adnan Siraj Rakin

**Abstract**: Recent advancements in side-channel attacks have revealed the vulnerability of modern Deep Neural Networks (DNNs) to malicious adversarial weight attacks. The well-studied RowHammer attack has effectively compromised DNN performance by inducing precise and deterministic bit-flips in the main memory (e.g., DRAM). Similarly, RowPress has emerged as another effective strategy for flipping targeted bits in DRAM. However, the impact of RowPress on deep learning applications has yet to be explored in the existing literature, leaving a fundamental research question unanswered: How does RowPress compare to RowHammer in leveraging bit-flip attacks to compromise DNN performance? This paper is the first to address this question and evaluate the impact of RowPress on DNN applications. We conduct a comparative analysis utilizing a novel DRAM-profile-aware attack designed to capture the distinct bit-flip patterns caused by RowHammer and RowPress. Eleven widely-used DNN architectures trained on different benchmark datasets deployed on a Samsung DRAM chip conclusively demonstrate that they suffer from a drastically more rapid performance degradation under the RowPress attack compared to RowHammer. The difference in the underlying attack mechanism of RowHammer and RowPress also renders existing RowHammer mitigation mechanisms ineffective under RowPress. As a result, RowPress introduces a new vulnerability paradigm for DNN compute platforms and unveils the urgent need for corresponding protective measures.



## **29. Dynamic Adversarial Attacks on Autonomous Driving Systems**

cs.RO

**SubmitDate**: 2024-12-03    [abs](http://arxiv.org/abs/2312.06701v3) [paper-pdf](http://arxiv.org/pdf/2312.06701v3)

**Authors**: Amirhosein Chahe, Chenan Wang, Abhishek Jeyapratap, Kaidi Xu, Lifeng Zhou

**Abstract**: This paper introduces an attacking mechanism to challenge the resilience of autonomous driving systems. Specifically, we manipulate the decision-making processes of an autonomous vehicle by dynamically displaying adversarial patches on a screen mounted on another moving vehicle. These patches are optimized to deceive the object detection models into misclassifying targeted objects, e.g., traffic signs. Such manipulation has significant implications for critical multi-vehicle interactions such as intersection crossing and lane changing, which are vital for safe and efficient autonomous driving systems. Particularly, we make four major contributions. First, we introduce a novel adversarial attack approach where the patch is not co-located with its target, enabling more versatile and stealthy attacks. Moreover, our method utilizes dynamic patches displayed on a screen, allowing for adaptive changes and movement, enhancing the flexibility and performance of the attack. To do so, we design a Screen Image Transformation Network (SIT-Net), which simulates environmental effects on the displayed images, narrowing the gap between simulated and real-world scenarios. Further, we integrate a positional loss term into the adversarial training process to increase the success rate of the dynamic attack. Finally, we shift the focus from merely attacking perceptual systems to influencing the decision-making algorithms of self-driving systems. Our experiments demonstrate the first successful implementation of such dynamic adversarial attacks in real-world autonomous driving scenarios, paving the way for advancements in the field of robust and secure autonomous driving.



## **30. Reactive Synthesis of Sensor Revealing Strategies in Hypergames on Graphs**

cs.GT

17 pages, 5 figures, 2 tables, submitted to Automatica

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01975v1) [paper-pdf](http://arxiv.org/pdf/2412.01975v1)

**Authors**: Sumukha Udupa, Ahmed Hemida, Charles A. Kamhoua, Jie Fu

**Abstract**: In many security applications of cyber-physical systems, a system designer must guarantee that critical missions are satisfied against attacks in the sensors and actuators of the CPS. Traditional security design of CPSs often assume that attackers have complete knowledge of the system. In this article, we introduce a class of deception techniques and study how to leverage asymmetric information created by deception to strengthen CPS security. Consider an adversarial interaction between a CPS defender and an attacker, who can perform sensor jamming attacks. To mitigate such attacks, the defender introduces asymmetrical information by deploying a "hidden sensor," whose presence is initially undisclosed but can be revealed if queried. We introduce hypergames on graphs to model this game with asymmetric information. Building on the solution concept called subjective rationalizable strategies in hypergames, we identify two stages in the game: An initial game stage where the defender commits to a strategy perceived rationalizable by the attacker until he deviates from the equilibrium in the attacker's perceptual game; Upon the deviation, a delay-attack game stage starts where the defender plays against the attacker, who has a bounded delay in attacking the sensor being revealed. Based on backward induction, we develop an algorithm that determines, for any given state, if the defender can benefit from hiding a sensor and revealing it later. If the answer is affirmative, the algorithm outputs a sensor revealing strategy to determine when to reveal the sensor during dynamic interactions. We demonstrate the effectiveness of our deceptive strategies through two case studies related to CPS security applications.



## **31. Topology-Based Reconstruction Prevention for Decentralised Learning**

cs.CR

14 pages, 19 figures, for associated experiment source code see  doi:10.4121/21572601.v2

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2312.05248v3) [paper-pdf](http://arxiv.org/pdf/2312.05248v3)

**Authors**: Florine W. Dekker, Zekeriya Erkin, Mauro Conti

**Abstract**: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed. To preserve the confidentiality of users' data, decentralised learning relies on differential privacy, multi-party computation, or both. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can infer other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adversary. The success rate depends only on the adversaries' direct neighbourhood, and is independent of the size of the full network. We consider weak adversaries that do not control the graph topology, cannot exploit the summation's inner workings, and do not have auxiliary knowledge; and show that these adversaries can still infer private data.   We analyse how reconstruction relates to topology and propose the first topology-based decentralised defence against reconstruction attacks. We show that reconstruction requires a number of adversaries linear in the length of the network's shortest cycle. Consequently, exact attacks over privacy-preserving summations are impossible in acyclic networks.   Our work is a stepping stone for a formal theory of topology-based decentralised reconstruction defences. Such a theory would generalise our countermeasure beyond summation, define confidentiality in terms of entropy, and describe the interactions with (topology-aware) differential privacy.



## **32. Effectiveness of L2 Regularization in Privacy-Preserving Machine Learning**

cs.LG

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01541v1) [paper-pdf](http://arxiv.org/pdf/2412.01541v1)

**Authors**: Nikolaos Chandrinos, Iliana Loi, Panagiotis Zachos, Ioannis Symeonidis, Aristotelis Spiliotis, Maria Panou, Konstantinos Moustakas

**Abstract**: Artificial intelligence, machine learning, and deep learning as a service have become the status quo for many industries, leading to the widespread deployment of models that handle sensitive data. Well-performing models, the industry seeks, usually rely on a large volume of training data. However, the use of such data raises serious privacy concerns due to the potential risks of leaks of highly sensitive information. One prominent threat is the Membership Inference Attack, where adversaries attempt to deduce whether a specific data point was used in a model's training process. An adversary's ability to determine an individual's presence represents a significant privacy threat, especially when related to a group of users sharing sensitive information. Hence, well-designed privacy-preserving machine learning solutions are critically needed in the industry. In this work, we compare the effectiveness of L2 regularization and differential privacy in mitigating Membership Inference Attack risks. Even though regularization techniques like L2 regularization are commonly employed to reduce overfitting, a condition that enhances the effectiveness of Membership Inference Attacks, their impact on mitigating these attacks has not been systematically explored.



## **33. Traversing the Subspace of Adversarial Patches**

cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01527v1) [paper-pdf](http://arxiv.org/pdf/2412.01527v1)

**Authors**: Jens Bayer, Stefan Becker, David Münch, Michael Arens, Jürgen Beyerer

**Abstract**: Despite ongoing research on the topic of adversarial examples in deep learning for computer vision, some fundamentals of the nature of these attacks remain unclear. As the manifold hypothesis posits, high-dimensional data tends to be part of a low-dimensional manifold. To verify the thesis with adversarial patches, this paper provides an analysis of a set of adversarial patches and investigates the reconstruction abilities of three different dimensionality reduction methods. Quantitatively, the performance of reconstructed patches in an attack setting is measured and the impact of sampled patches from the latent space during adversarial training is investigated. The evaluation is performed on two publicly available datasets for person detection. The results indicate that more sophisticated dimensionality reduction methods offer no advantages over a simple principal component analysis.



## **34. Adversarial Attacks on Hyperbolic Networks**

cs.LG

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01495v1) [paper-pdf](http://arxiv.org/pdf/2412.01495v1)

**Authors**: Max van Spengler, Jan Zahálka, Pascal Mettes

**Abstract**: As hyperbolic deep learning grows in popularity, so does the need for adversarial robustness in the context of such a non-Euclidean geometry. To this end, this paper proposes hyperbolic alternatives to the commonly used FGM and PGD adversarial attacks. Through interpretable synthetic benchmarks and experiments on existing datasets, we show how the existing and newly proposed attacks differ. Moreover, we investigate the differences in adversarial robustness between Euclidean and fully hyperbolic networks. We find that these networks suffer from different types of vulnerabilities and that the newly proposed hyperbolic attacks cannot address these differences. Therefore, we conclude that the shifts in adversarial robustness are due to the models learning distinct patterns resulting from their different geometries.



## **35. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01440v1) [paper-pdf](http://arxiv.org/pdf/2412.01440v1)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.



## **36. Behavior Backdoor for Deep Learning Models**

cs.LG

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01369v1) [paper-pdf](http://arxiv.org/pdf/2412.01369v1)

**Authors**: Jiakai Wang, Pengfei Zhang, Renshuai Tao, Jian Yang, Hao Liu, Xianglong Liu, Yunchao Wei, Yao Zhao

**Abstract**: The various post-processing methods for deep-learning-based models, such as quantification, pruning, and fine-tuning, play an increasingly important role in artificial intelligence technology, with pre-train large models as one of the main development directions. However, this popular series of post-processing behaviors targeting pre-training deep models has become a breeding ground for new adversarial security issues. In this study, we take the first step towards ``behavioral backdoor'' attack, which is defined as a behavior-triggered backdoor model training procedure, to reveal a new paradigm of backdoor attacks. In practice, we propose the first pipeline of implementing behavior backdoor, i.e., the Quantification Backdoor (QB) attack, upon exploiting model quantification method as the set trigger. Specifically, to adapt the optimization goal of behavior backdoor, we introduce the behavior-driven backdoor object optimizing method by a bi-target behavior backdoor training loss, thus we could guide the poisoned model optimization direction. To update the parameters across multiple models, we adopt the address-shared backdoor model training, thereby the gradient information could be utilized for multimodel collaborative optimization. Extensive experiments have been conducted on different models, datasets, and tasks, demonstrating the effectiveness of this novel backdoor attack and its potential application threats.



## **37. Exploring the Robustness of AI-Driven Tools in Digital Forensics: A Preliminary Study**

cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01363v1) [paper-pdf](http://arxiv.org/pdf/2412.01363v1)

**Authors**: Silvia Lucia Sanna, Leonardo Regano, Davide Maiorca, Giorgio Giacinto

**Abstract**: Nowadays, many tools are used to facilitate forensic tasks about data extraction and data analysis. In particular, some tools leverage Artificial Intelligence (AI) to automatically label examined data into specific categories (\ie, drugs, weapons, nudity). However, this raises a serious concern about the robustness of the employed AI algorithms against adversarial attacks. Indeed, some people may need to hide specific data to AI-based digital forensics tools, thus manipulating the content so that the AI system does not recognize the offensive/prohibited content and marks it at as suspicious to the analyst. This could be seen as an anti-forensics attack scenario. For this reason, we analyzed two of the most important forensics tools employing AI for data classification: Magnet AI, used by Magnet Axiom, and Excire Photo AI, used by X-Ways Forensics. We made preliminary tests using about $200$ images, other $100$ sent in $3$ chats about pornography and teenage nudity, drugs and weapons to understand how the tools label them. Moreover, we loaded some deepfake images (images generated by AI forging real ones) of some actors to understand if they would be classified in the same category as the original images. From our preliminary study, we saw that the AI algorithm is not robust enough, as we expected since these topics are still open research problems. For example, some sexual images were not categorized as nudity, and some deepfakes were categorized as the same real person, while the human eye can see the clear nudity image or catch the difference between the deepfakes. Building on these results and other state-of-the-art works, we provide some suggestions for improving how digital forensics analysis tool leverage AI and their robustness against adversarial attacks or different scenarios than the trained one.



## **38. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

8 pages, 2 figures

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2410.03218v3) [paper-pdf](http://arxiv.org/pdf/2410.03218v3)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are arbitrary. We show that when the probability of having an attack at a given time is less than 0.5 and each attack spans the entire space in expectation, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **39. Data-Driven and Stealthy Deactivation of Safety Filters**

eess.SY

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01346v1) [paper-pdf](http://arxiv.org/pdf/2412.01346v1)

**Authors**: Daniel Arnström, André M. H. Teixeira

**Abstract**: Safety filters ensure that control actions that are executed are always safe, no matter the controller in question. Previous work has proposed a simple and stealthy false-data injection attack for deactivating such safety filters. This attack injects false sensor measurements to bias state estimates toward the interior of a safety region, making the safety filter accept unsafe control actions. The attack does, however, require the adversary to know the dynamics of the system, the safety region used in the safety filter, and the observer gain. In this work we relax these requirements and show how a similar data-injection attack can be performed when the adversary only observes the input and output of the observer that is used by the safety filter, without any a priori knowledge about the system dynamics, safety region, or observer gain. In particular, the adversary uses the observed data to identify a state-space model that describes the observer dynamics, and then approximates a safety region in the identified embedding. We exemplify the data-driven attack on an inverted pendulum, where we show how the attack can make the system leave a safe set, even when a safety filter is supposed to stop this from happening.



## **40. CantorNet: A Sandbox for Testing Geometrical and Topological Complexity Measures**

cs.NE

Accepted at the NeurIPS Workshop on Symmetry and Geometry in Neural  Representations, 2024

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2411.19713v2) [paper-pdf](http://arxiv.org/pdf/2411.19713v2)

**Authors**: Michal Lewandowski, Hamid Eghbalzadeh, Bernhard A. Moser

**Abstract**: Many natural phenomena are characterized by self-similarity, for example the symmetry of human faces, or a repetitive motif of a song. Studying of such symmetries will allow us to gain deeper insights into the underlying mechanisms of complex systems. Recognizing the importance of understanding these patterns, we propose a geometrically inspired framework to study such phenomena in artificial neural networks. To this end, we introduce \emph{CantorNet}, inspired by the triadic construction of the Cantor set, which was introduced by Georg Cantor in the $19^\text{th}$ century. In mathematics, the Cantor set is a set of points lying on a single line that is self-similar and has a counter intuitive property of being an uncountably infinite null set. Similarly, we introduce CantorNet as a sandbox for studying self-similarity by means of novel topological and geometrical complexity measures. CantorNet constitutes a family of ReLU neural networks that spans the whole spectrum of possible Kolmogorov complexities, including the two opposite descriptions (linear and exponential as measured by the description length). CantorNet's decision boundaries can be arbitrarily ragged, yet are analytically known. Besides serving as a testing ground for complexity measures, our work may serve to illustrate potential pitfalls in geometry-ignorant data augmentation techniques and adversarial attacks.



## **41. Hiding Faces in Plain Sight: Defending DeepFakes by Disrupting Face Detection**

cs.CV

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2412.01101v1) [paper-pdf](http://arxiv.org/pdf/2412.01101v1)

**Authors**: Delong Zhu, Yuezun Li, Baoyuan Wu, Jiaran Zhou, Zhibo Wang, Siwei Lyu

**Abstract**: This paper investigates the feasibility of a proactive DeepFake defense framework, {\em FacePosion}, to prevent individuals from becoming victims of DeepFake videos by sabotaging face detection. The motivation stems from the reliance of most DeepFake methods on face detectors to automatically extract victim faces from videos for training or synthesis (testing). Once the face detectors malfunction, the extracted faces will be distorted or incorrect, subsequently disrupting the training or synthesis of the DeepFake model. To achieve this, we adapt various adversarial attacks with a dedicated design for this purpose and thoroughly analyze their feasibility. Based on FacePoison, we introduce {\em VideoFacePoison}, a strategy that propagates FacePoison across video frames rather than applying them individually to each frame. This strategy can largely reduce the computational overhead while retaining the favorable attack performance. Our method is validated on five face detectors, and extensive experiments against eleven different DeepFake models demonstrate the effectiveness of disrupting face detectors to hinder DeepFake generation.



## **42. OffRAMPS: An FPGA-based Intermediary for Analysis and Modification of Additive Manufacturing Control Systems**

cs.CR

**SubmitDate**: 2024-12-02    [abs](http://arxiv.org/abs/2404.15446v2) [paper-pdf](http://arxiv.org/pdf/2404.15446v2)

**Authors**: Jason Blocklove, Md Raz, Prithwish Basu Roy, Hammond Pearce, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Cybersecurity threats in Additive Manufacturing (AM) are an increasing concern as AM adoption continues to grow. AM is now being used for parts in the aerospace, transportation, and medical domains. Threat vectors which allow for part compromise are particularly concerning, as any failure in these domains would have life-threatening consequences. A major challenge to investigation of AM part-compromises comes from the difficulty in evaluating and benchmarking both identified threat vectors as well as methods for detecting adversarial actions. In this work, we introduce a generalized platform for systematic analysis of attacks against and defenses for 3D printers. Our "OFFRAMPS" platform is based on the open-source 3D printer control board "RAMPS." OFFRAMPS allows analysis, recording, and modification of all control signals and I/O for a 3D printer. We show the efficacy of OFFRAMPS by presenting a series of case studies based on several Trojans, including ones identified in the literature, and show that OFFRAMPS can both emulate and detect these attacks, i.e., it can both change and detect arbitrary changes to the g-code print commands.



## **43. Online Poisoning Attack Against Reinforcement Learning under Black-box Environments**

cs.LG

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00797v1) [paper-pdf](http://arxiv.org/pdf/2412.00797v1)

**Authors**: Jianhui Li, Bokang Zhang, Junfeng Wu

**Abstract**: This paper proposes an online environment poisoning algorithm tailored for reinforcement learning agents operating in a black-box setting, where an adversary deliberately manipulates training data to lead the agent toward a mischievous policy. In contrast to prior studies that primarily investigate white-box settings, we focus on a scenario characterized by \textit{unknown} environment dynamics to the attacker and a \textit{flexible} reinforcement learning algorithm employed by the targeted agent. We first propose an attack scheme that is capable of poisoning the reward functions and state transitions. The poisoning task is formalized as a constrained optimization problem, following the framework of \cite{ma2019policy}. Given the transition probabilities are unknown to the attacker in a black-box environment, we apply a stochastic gradient descent algorithm, where the exact gradients are approximated using sample-based estimates. A penalty-based method along with a bilevel reformulation is then employed to transform the problem into an unconstrained counterpart and to circumvent the double-sampling issue. The algorithm's effectiveness is validated through a maze environment.



## **44. Learning to Forget using Hypernetworks**

cs.LG

AdvML-Frontiers'24: The 3rd Workshop on New Frontiers in Adversarial  Machine Learning@NeurIPS'24, Vancouver, CA

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00761v1) [paper-pdf](http://arxiv.org/pdf/2412.00761v1)

**Authors**: Jose Miguel Lara Rangel, Stefan Schoepf, Jack Foster, David Krueger, Usman Anwar

**Abstract**: Machine unlearning is gaining increasing attention as a way to remove adversarial data poisoning attacks from already trained models and to comply with privacy and AI regulations. The objective is to unlearn the effect of undesired data from a trained model while maintaining performance on the remaining data. This paper introduces HyperForget, a novel machine unlearning framework that leverages hypernetworks - neural networks that generate parameters for other networks - to dynamically sample models that lack knowledge of targeted data while preserving essential capabilities. Leveraging diffusion models, we implement two Diffusion HyperForget Networks and used them to sample unlearned models in Proof-of-Concept experiments. The unlearned models obtained zero accuracy on the forget set, while preserving good accuracy on the retain sets, highlighting the potential of HyperForget for dynamic targeted data removal and a promising direction for developing adaptive machine unlearning algorithms.



## **45. Intermediate Outputs Are More Sensitive Than You Think**

cs.CV

**SubmitDate**: 2024-12-01    [abs](http://arxiv.org/abs/2412.00696v1) [paper-pdf](http://arxiv.org/pdf/2412.00696v1)

**Authors**: Tao Huang, Qingyu Huang, Jiayang Meng

**Abstract**: The increasing reliance on deep computer vision models that process sensitive data has raised significant privacy concerns, particularly regarding the exposure of intermediate results in hidden layers. While traditional privacy risk assessment techniques focus on protecting overall model outputs, they often overlook vulnerabilities within these intermediate representations. Current privacy risk assessment techniques typically rely on specific attack simulations to assess risk, which can be computationally expensive and incomplete. This paper introduces a novel approach to measuring privacy risks in deep computer vision models based on the Degrees of Freedom (DoF) and sensitivity of intermediate outputs, without requiring adversarial attack simulations. We propose a framework that leverages DoF to evaluate the amount of information retained in each layer and combines this with the rank of the Jacobian matrix to assess sensitivity to input variations. This dual analysis enables systematic measurement of privacy risks at various model layers. Our experimental validation on real-world datasets demonstrates the effectiveness of this approach in providing deeper insights into privacy risks associated with intermediate representations.



## **46. Exact Certification of (Graph) Neural Networks Against Label Poisoning**

cs.LG

Under review

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00537v1) [paper-pdf](http://arxiv.org/pdf/2412.00537v1)

**Authors**: Mahalakshmi Sabanayagam, Lukas Gosch, Stephan Günnemann, Debarghya Ghoshdastidar

**Abstract**: Machine learning models are highly vulnerable to label flipping, i.e., the adversarial modification (poisoning) of training labels to compromise performance. Thus, deriving robustness certificates is important to guarantee that test predictions remain unaffected and to understand worst-case robustness behavior. However, for Graph Neural Networks (GNNs), the problem of certifying label flipping has so far been unsolved. We change this by introducing an exact certification method, deriving both sample-wise and collective certificates. Our method leverages the Neural Tangent Kernel (NTK) to capture the training dynamics of wide networks enabling us to reformulate the bilevel optimization problem representing label flipping into a Mixed-Integer Linear Program (MILP). We apply our method to certify a broad range of GNN architectures in node classification tasks. Thereby, concerning the worst-case robustness to label flipping: $(i)$ we establish hierarchies of GNNs on different benchmark graphs; $(ii)$ quantify the effect of architectural choices such as activations, depth and skip-connections; and surprisingly, $(iii)$ uncover a novel phenomenon of the robustness plateauing for intermediate perturbation budgets across all investigated datasets and architectures. While we focus on GNNs, our certificates are applicable to sufficiently wide NNs in general through their NTK. Thus, our work presents the first exact certificate to a poisoning attack ever derived for neural networks, which could be of independent interest.



## **47. Hard-Label Black-Box Attacks on 3D Point Clouds**

cs.CV

**SubmitDate**: 2024-11-30    [abs](http://arxiv.org/abs/2412.00404v1) [paper-pdf](http://arxiv.org/pdf/2412.00404v1)

**Authors**: Daizong Liu, Yunbo Tao, Pan Zhou, Wei Hu

**Abstract**: With the maturity of depth sensors in various 3D safety-critical applications, 3D point cloud models have been shown to be vulnerable to adversarial attacks. Almost all existing 3D attackers simply follow the white-box or black-box setting to iteratively update coordinate perturbations based on back-propagated or estimated gradients. However, these methods are hard to deploy in real-world scenarios (no model details are provided) as they severely rely on parameters or output logits of victim models. To this end, we propose point cloud attacks from a more practical setting, i.e., hard-label black-box attack, in which attackers can only access the prediction label of 3D input. We introduce a novel 3D attack method based on a new spectrum-aware decision boundary algorithm to generate high-quality adversarial samples. In particular, we first construct a class-aware model decision boundary, by developing a learnable spectrum-fusion strategy to adaptively fuse point clouds of different classes in the spectral domain, aiming to craft their intermediate samples without distorting the original geometry. Then, we devise an iterative coordinate-spectrum optimization method with curvature-aware boundary search to move the intermediate sample along the decision boundary for generating adversarial point clouds with trivial perturbations. Experiments demonstrate that our attack competitively outperforms existing white/black-box attackers in terms of attack performance and adversary quality.



## **48. Calibration Attacks: A Comprehensive Study of Adversarial Attacks on Model Confidence**

cs.LG

Accepted at Transactions on Machine Learning Research

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2401.02718v3) [paper-pdf](http://arxiv.org/pdf/2401.02718v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu, Hongyu Guo

**Abstract**: In this work, we highlight and perform a comprehensive study on calibration attacks, a form of adversarial attacks that aim to trap victim models to be heavily miscalibrated without altering their predicted labels, hence endangering the trustworthiness of the models and follow-up decision making based on their confidence. We propose four typical forms of calibration attacks: underconfidence, overconfidence, maximum miscalibration, and random confidence attacks, conducted in both black-box and white-box setups. We demonstrate that the attacks are highly effective on both convolutional and attention-based models: with a small number of queries, they seriously skew confidence without changing the predictive performance. Given the potential danger, we further investigate the effectiveness of a wide range of adversarial defence and recalibration methods, including our proposed defences specifically designed for calibration attacks to mitigate the harm. From the ECE and KS scores, we observe that there are still significant limitations in handling calibration attacks. To the best of our knowledge, this is the first dedicated study that provides a comprehensive investigation on calibration-focused attacks. We hope this study helps attract more attention to these types of attacks and hence hamper their potential serious damages. To this end, this work also provides detailed analyses to understand the characteristics of the attacks. Our code is available at https://github.com/PhenetOs/CalibrationAttack



## **49. Towards Class-wise Robustness Analysis**

cs.LG

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2411.19853v1) [paper-pdf](http://arxiv.org/pdf/2411.19853v1)

**Authors**: Tejaswini Medi, Julia Grabinski, Margret Keuper

**Abstract**: While being very successful in solving many downstream tasks, the application of deep neural networks is limited in real-life scenarios because of their susceptibility to domain shifts such as common corruptions, and adversarial attacks. The existence of adversarial examples and data corruption significantly reduces the performance of deep classification models. Researchers have made strides in developing robust neural architectures to bolster decisions of deep classifiers. However, most of these works rely on effective adversarial training methods, and predominantly focus on overall model robustness, disregarding class-wise differences in robustness, which are critical. Exploiting weakly robust classes is a potential avenue for attackers to fool the image recognition models. Therefore, this study investigates class-to-class biases across adversarially trained robust classification models to understand their latent space structures and analyze their strong and weak class-wise properties. We further assess the robustness of classes against common corruptions and adversarial attacks, recognizing that class vulnerability extends beyond the number of correct classifications for a specific class. We find that the number of false positives of classes as specific target classes significantly impacts their vulnerability to attacks. Through our analysis on the Class False Positive Score, we assess a fair evaluation of how susceptible each class is to misclassification.



## **50. ModSec-AdvLearn: Countering Adversarial SQL Injections with Robust Machine Learning**

cs.LG

**SubmitDate**: 2024-11-29    [abs](http://arxiv.org/abs/2308.04964v3) [paper-pdf](http://arxiv.org/pdf/2308.04964v3)

**Authors**: Biagio Montaruli, Giuseppe Floris, Christian Scano, Luca Demetrio, Andrea Valenza, Luca Compagna, Davide Ariu, Luca Piras, Davide Balzarotti, Battista Biggio

**Abstract**: Many Web Application Firewalls (WAFs) leverage the OWASP Core Rule Set (CRS) to block incoming malicious requests. The CRS consists of different sets of rules designed by domain experts to detect well-known web attack patterns. Both the set of rules to be used and the weights used to combine them are manually defined, yielding four different default configurations of the CRS. In this work, we focus on the detection of SQL injection (SQLi) attacks, and show that the manual configurations of the CRS typically yield a suboptimal trade-off between detection and false alarm rates. Furthermore, we show that these configurations are not robust to adversarial SQLi attacks, i.e., carefully-crafted attacks that iteratively refine the malicious SQLi payload by querying the target WAF to bypass detection. To overcome these limitations, we propose (i) using machine learning to automate the selection of the set of rules to be combined along with their weights, i.e., customizing the CRS configuration based on the monitored web services; and (ii) leveraging adversarial training to significantly improve its robustness to adversarial SQLi manipulations. Our experiments, conducted using the well-known open-source ModSecurity WAF equipped with the CRS rules, show that our approach, named ModSec-AdvLearn, can (i) increase the detection rate up to 30%, while retaining negligible false alarm rates and discarding up to 50% of the CRS rules; and (ii) improve robustness against adversarial SQLi attacks up to 85%, marking a significant stride toward designing more effective and robust WAFs. We release our open-source code at https://github.com/pralab/modsec-advlearn.



