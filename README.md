# Latest Adversarial Attack Papers
**update at 2025-02-12 10:58:18**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

math.OC

20 pages, 1 figure

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07603v1) [paper-pdf](http://arxiv.org/pdf/2502.07603v1)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting considers only simple linear models and not general nonlinear dynamical systems. We first characterize the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Based on this approximation, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A simulation example on an academic nonlinear system demonstrates that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies.



## **2. Efficient Image-to-Image Diffusion Classifier for Adversarial Robustness**

cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2408.08502v2) [paper-pdf](http://arxiv.org/pdf/2408.08502v2)

**Authors**: Hefei Mei, Minjing Dong, Chang Xu

**Abstract**: Diffusion models (DMs) have demonstrated great potential in the field of adversarial robustness, where DM-based defense methods can achieve superior defense capability without adversarial training. However, they all require huge computational costs due to the usage of large-scale pre-trained DMs, making it difficult to conduct full evaluation under strong attacks and compare with traditional CNN-based methods. Simply reducing the network size and timesteps in DMs could significantly harm the image generation quality, which invalidates previous frameworks. To alleviate this issue, we redesign the diffusion framework from generating high-quality images to predicting distinguishable image labels. Specifically, we employ an image translation framework to learn many-to-one mapping from input samples to designed orthogonal image labels. Based on this framework, we introduce an efficient Image-to-Image diffusion classifier with a pruned U-Net structure and reduced diffusion timesteps. Besides the framework, we redesign the optimization objective of DMs to fit the target of image classification, where a new classification loss is incorporated in the DM-based image translation framework to distinguish the generated label from those of other classes. We conduct sufficient evaluations of the proposed classifier under various attacks on popular benchmarks. Extensive experiments show that our method achieves better adversarial robustness with fewer computational costs than DM-based and CNN-based methods. The code is available at https://github.com/hfmei/IDC



## **3. RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization**

cs.CR

13 pages, 4 figures

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07492v1) [paper-pdf](http://arxiv.org/pdf/2502.07492v1)

**Authors**: Yuxia Sun, Huihong Chen, Jingcai Guo, Aoxiang Sun, Zhetao Li, Haolin Liu

**Abstract**: Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adversaries often conceal their identities, rendering attribution inherently adversarial. Existing machine learning-based attribution models, while effective, remain highly vulnerable to adversarial attacks. For example, the state-of-the-art byte-level model MalConv sees its accuracy drop from over 90% to below 2% under PGD (projected gradient descent) attacks. Existing gradient-based adversarial training techniques for malware detection or image processing were applied to malware attribution in this study, revealing that both robustness and training efficiency require significant improvement. To address this, we propose RoMA, a novel single-step adversarial training approach that integrates global perturbations to generate enhanced adversarial samples and employs adversarial consistency regularization to improve representation quality and resilience. A novel APT malware dataset named AMG18, with diverse samples and realistic class imbalances, is introduced for evaluation. Extensive experiments show that RoMA significantly outperforms seven competing methods in both adversarial robustness (e.g., achieving over 80% robust accuracy-more than twice that of the next-best method under PGD attacks) and training efficiency (e.g., more than twice as fast as the second-best method in terms of accuracy), while maintaining superior standard accuracy in non-adversarial scenarios.



## **4. Mining Power Destruction Attacks in the Presence of Petty-Compliant Mining Pools**

cs.CR

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07410v1) [paper-pdf](http://arxiv.org/pdf/2502.07410v1)

**Authors**: Roozbeh Sarenche, Svetla Nikova, Bart Preneel

**Abstract**: Bitcoin's security relies on its Proof-of-Work consensus, where miners solve puzzles to propose blocks. The puzzle's difficulty is set by the difficulty adjustment mechanism (DAM), based on the network's available mining power. Attacks that destroy some portion of mining power can exploit the DAM to lower difficulty, making such attacks profitable. In this paper, we analyze three types of mining power destruction attacks in the presence of petty-compliant mining pools: selfish mining, bribery, and mining power distraction attacks. We analyze selfish mining while accounting for the distribution of mining power among pools, a factor often overlooked in the literature. Our findings indicate that selfish mining can be more destructive when the non-adversarial mining share is well distributed among pools. We also introduce a novel bribery attack, where the adversarial pool bribes petty-compliant pools to orphan others' blocks. For small pools, we demonstrate that the bribery attack can dominate strategies like selfish mining or undercutting. Lastly, we present the mining distraction attack, where the adversarial pool incentivizes petty-compliant pools to abandon Bitcoin's puzzle and mine for a simpler puzzle, thus wasting some part of their mining power. Similar to the previous attacks, this attack can lower the mining difficulty, but with the difference that it does not generate any evidence of mining power destruction, such as orphan blocks.



## **5. Enhancing Security and Privacy in Federated Learning using Low-Dimensional Update Representation and Proximity-Based Defense**

cs.CR

14 pages

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2405.18802v2) [paper-pdf](http://arxiv.org/pdf/2405.18802v2)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, particularly against curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{F}ederated \underline{L}earning with Low-Dimensional \underline{U}pdate \underline{R}epresentation and \underline{P}roximity-Based defense (FLURP), designed to address privacy preservation and resistance to Byzantine attacks in distributed learning environments. FLURP employs $\mathsf{LinfSample}$ method, enabling clients to compute the $l_{\infty}$ norm across sliding windows of updates, resulting in a Low-Dimensional Update Representation (LUR). Calculating the shared distance matrix among LURs, rather than updates, significantly reduces the overhead of Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and poisoned updates. Additionally, FLURP integrates a privacy-preserving proximity-based defense mechanism utilizing optimized SMPC protocols to minimize communication rounds. Our experiments demonstrate FLURP's effectiveness in countering Byzantine adversaries with low communication and runtime overhead. FLURP offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.



## **6. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models**

cs.CV

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07225v1) [paper-pdf](http://arxiv.org/pdf/2502.07225v1)

**Authors**: Sen Peng, Mingyue Wang, Jianfei He, Jijia Yang, Xiaohua Jia

**Abstract**: Latent diffusion models have recently demonstrated superior capabilities in many downstream image synthesis tasks. However, customization of latent diffusion models using unauthorized data can severely compromise the privacy and intellectual property rights of data owners. Adversarial examples as protective perturbations have been developed to defend against unauthorized data usage by introducing imperceptible noise to customization samples, preventing diffusion models from effectively learning them. In this paper, we first reveal that the primary reason adversarial examples are effective as protective perturbations in latent diffusion models is the distortion of their latent representations, as demonstrated through qualitative and quantitative experiments. We then propose the Contrastive Adversarial Training (CAT) utilizing adapters as an adaptive attack against these protection methods, highlighting their lack of robustness. Extensive experiments demonstrate that our CAT method significantly reduces the effectiveness of protective perturbations in customization configurations, urging the community to reconsider and enhance the robustness of existing protective perturbation methods. Code is available at \hyperlink{here}{https://github.com/senp98/CAT}.



## **7. LUNAR: LLM Unlearning via Neural Activation Redirection**

cs.LG

**SubmitDate**: 2025-02-11    [abs](http://arxiv.org/abs/2502.07218v1) [paper-pdf](http://arxiv.org/pdf/2502.07218v1)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.



## **8. SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07101v1) [paper-pdf](http://arxiv.org/pdf/2502.07101v1)

**Authors**: Saurabh Kumar Pandey, Sachin Vashistha, Debrup Das, Somak Aditya, Monojit Choudhury

**Abstract**: To understand the complexity of sequence classification tasks, Hahn et al. (2021) proposed sensitivity as the number of disjoint subsets of the input sequence that can each be individually changed to change the output. Though effective, calculating sensitivity at scale using this framework is costly because of exponential time complexity. Therefore, we introduce a Sensitivity-based Multi-Armed Bandit framework (SMAB), which provides a scalable approach for calculating word-level local (sentence-level) and global (aggregated) sensitivities concerning an underlying text classifier for any dataset. We establish the effectiveness of our approach through various applications. We perform a case study on CHECKLIST generated sentiment analysis dataset where we show that our algorithm indeed captures intuitively high and low-sensitive words. Through experiments on multiple tasks and languages, we show that sensitivity can serve as a proxy for accuracy in the absence of gold data. Lastly, we show that guiding perturbation prompts using sensitivity values in adversarial example generation improves attack success rate by 15.58%, whereas using sensitivity as an additional reward in adversarial paraphrase generation gives a 12.00% improvement over SOTA approaches. Warning: Contains potentially offensive content.



## **9. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

cs.LG

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.07011v1) [paper-pdf](http://arxiv.org/pdf/2502.07011v1)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.



## **10. Breaking Quantum Key Distributions under Quantum Switch-Based Attack**

quant-ph

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06780v1) [paper-pdf](http://arxiv.org/pdf/2502.06780v1)

**Authors**: Sumit Nandi, Biswaranjan Panda, Pankaj Agrawal, Arun K Pati

**Abstract**: Quantum key distribution (QKD) enables secure key sharing between distant parties, with several protocols proven resilient against conventional eavesdropping strategies. Here, we introduce a new attack scenario where an eavesdropper, Eve, exploits a quantum switch using the indefinite causal order to intercept and manipulate quantum communication channel. Using multiple metrics such as the information gain, mutual information, and Bell violation, we demonstrate that the presence of a quantum switch significantly compromises QKD security. Our results highlight a previously overlooked vulnerability, emphasizing the need for countermeasures against quantum-controlled adversarial strategies.



## **11. When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning**

cs.LG

Accepted at AAAI 2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2409.14161v3) [paper-pdf](http://arxiv.org/pdf/2409.14161v3)

**Authors**: Naheed Anjum Arafat, Debabrota Basu, Yulia Gel, Yuzhou Chen

**Abstract**: Capitalizing on the intuitive premise that shape characteristics are more robust to perturbations, we bridge adversarial graph learning with the emerging tools from computational topology, namely, persistent homology representations of graphs. We introduce the concept of witness complex to adversarial analysis on graphs, which allows us to focus only on the salient shape characteristics of graphs, yielded by the subset of the most essential nodes (i.e., landmarks), with minimal loss of topological information on the whole graph. The remaining nodes are then used as witnesses, governing which higher-order graph substructures are incorporated into the learning process. Armed with the witness mechanism, we design Witness Graph Topological Layer (WGTL), which systematically integrates both local and global topological graph feature representations, the impact of which is, in turn, automatically controlled by the robust regularized topological loss. Given the attacker's budget, we derive the important stability guarantees of both local and global topology encodings and the associated robust topological loss. We illustrate the versatility and efficiency of WGTL by its integration with five GNNs and three existing non-topological defense mechanisms. Our extensive experiments across six datasets demonstrate that WGTL boosts the robustness of GNNs across a range of perturbations and against a range of adversarial attacks. Our datasets and source codes are available at https://github.com/toggled/WGTL.



## **12. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2408.00761v4) [paper-pdf](http://arxiv.org/pdf/2408.00761v4)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **13. Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks**

cs.CR

Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop  on Privacy-Preserving Artificial Intelligence

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2501.18727v2) [paper-pdf](http://arxiv.org/pdf/2501.18727v2)

**Authors**: Mohd. Farhan Israk Soumik, W. K. M. Mithsara, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.



## **14. LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

cs.CL

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.05232v2) [paper-pdf](http://arxiv.org/pdf/2412.05232v2)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Traditional jailbreaks have successfully exposed vulnerabilities in LLMs, primarily relying on discrete combinatorial optimization, while more recent methods focus on training LLMs to generate adversarial prompts. However, both approaches are computationally expensive and slow, often requiring significant resources to generate a single successful attack. We hypothesize that the inefficiency of these methods arises from an inadequate characterization of the jailbreak problem itself. To address this gap, we approach the jailbreak problem as an alignment problem, leading us to propose LIAR (Leveraging Inference time Alignment to jailbReak), a fast and efficient best-of-N approach tailored for jailbreak attacks. LIAR offers several key advantages: it eliminates the need for additional training, operates in a fully black-box setting, significantly reduces computational overhead, and produces more human-readable adversarial prompts while maintaining competitive attack success rates. Our results demonstrate that a best-of-N approach is a simple yet highly effective strategy for evaluating the robustness of aligned LLMs, achieving attack success rates (ASR) comparable to state-of-the-art methods while offering a 10x improvement in perplexity and a significant speedup in Time-to-Attack, reducing execution time from tens of hours to seconds. Additionally, We also provide sub-optimality guarantees for the proposed LIAR. Our work highlights the potential of efficient, alignment-based jailbreak strategies for assessing and stress-testing AI safety measures.



## **15. Automatic ISA analysis for Secure Context Switching**

cs.OS

15 pages, 6 figures, 2 tables, 4 listings

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06609v1) [paper-pdf](http://arxiv.org/pdf/2502.06609v1)

**Authors**: Neelu S. Kalani, Thomas Bourgeat, Guerney D. H. Hunt, Wojciech Ozga

**Abstract**: Instruction set architectures are complex, with hundreds of registers and instructions that can modify dozens of them during execution, variably on each instance. Prose-style ISA specifications struggle to capture these intricacies of the ISAs, where often the important details about a single register are spread out across hundreds of pages of documentation. Ensuring that all ISA-state is swapped in context switch implementations of privileged software requires meticulous examination of these pages. This manual process is tedious and error-prone.   We propose a tool called Sailor that leverages machine-readable ISA specifications written in Sail to automate this task. Sailor determines the ISA-state necessary to swap during the context switch using the data collected from Sail and a novel algorithm to classify ISA-state as security-sensitive. Using Sailor's output, we identify three different classes of mishandled ISA-state across four open-source confidential computing systems. We further reveal five distinct security vulnerabilities that can be exploited using the mishandled ISA-state. This research exposes an often overlooked attack surface that stems from mishandled ISA-state, enabling unprivileged adversaries to exploit system vulnerabilities.



## **16. Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning**

cs.LG

Submitted to Neural Networks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06917v1) [paper-pdf](http://arxiv.org/pdf/2502.06917v1)

**Authors**: Mario García-Márquez, Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: Federated Learning presents a nascent approach to machine learning, enabling collaborative model training across decentralized devices while safeguarding data privacy. However, its distributed nature renders it susceptible to adversarial attacks. Integrating blockchain technology with Federated Learning offers a promising avenue to enhance security and integrity. In this paper, we tackle the potential of blockchain in defending Federated Learning against adversarial attacks. First, we test Proof of Federated Learning, a well known consensus mechanism designed ad-hoc to federated contexts, as a defense mechanism demonstrating its efficacy against Byzantine and backdoor attacks when at least one miner remains uncompromised. Second, we propose Krum Federated Chain, a novel defense strategy combining Krum and Proof of Federated Learning, valid to defend against any configuration of Byzantine or backdoor attacks, even when all miners are compromised. Our experiments conducted on image classification datasets validate the effectiveness of our proposed approaches.



## **17. Robust Watermarks Leak: Channel-Aware Feature Extraction Enables Adversarial Watermark Manipulation**

cs.CV

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06418v1) [paper-pdf](http://arxiv.org/pdf/2502.06418v1)

**Authors**: Zhongjie Ba, Yitao Zhang, Peng Cheng, Bin Gong, Xinyu Zhang, Qinglong Wang, Kui Ren

**Abstract**: Watermarking plays a key role in the provenance and detection of AI-generated content. While existing methods prioritize robustness against real-world distortions (e.g., JPEG compression and noise addition), we reveal a fundamental tradeoff: such robust watermarks inherently improve the redundancy of detectable patterns encoded into images, creating exploitable information leakage. To leverage this, we propose an attack framework that extracts leakage of watermark patterns through multi-channel feature learning using a pre-trained vision model. Unlike prior works requiring massive data or detector access, our method achieves both forgery and detection evasion with a single watermarked image. Extensive experiments demonstrate that our method achieves a 60\% success rate gain in detection evasion and 51\% improvement in forgery accuracy compared to state-of-the-art methods while maintaining visual fidelity. Our work exposes the robustness-stealthiness paradox: current "robust" watermarks sacrifice security for distortion resistance, providing insights for future watermark design.



## **18. Hyperparameters in Score-Based Membership Inference Attacks**

cs.LG

This work has been accepted for publication in the 3rd IEEE  Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final  version will be available on IEEE Xplore

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.06374v1) [paper-pdf](http://arxiv.org/pdf/2502.06374v1)

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA.



## **19. TASAR: Transfer-based Attack on Skeletal Action Recognition**

cs.CV

arXiv admin note: text overlap with arXiv:2407.08572

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2409.02483v4) [paper-pdf](http://arxiv.org/pdf/2409.02483v4)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xiaoshuai Hao, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequences, as well-structured representations of human behaviors, play a vital role in Human Activity Recognition (HAR). The transferability of adversarial skeletal sequences enables attacks in real-world HAR scenarios, such as autonomous driving, intelligent surveillance, and human-computer interactions. However, most existing skeleton-based HAR (S-HAR) attacks are primarily designed for white-box scenarios and exhibit weak adversarial transferability. Therefore, they cannot be considered true transfer-based S-HAR attacks. More importantly, the reason for this failure remains unclear. In this paper, we study this phenomenon through the lens of loss surface, and find that its sharpness contributes to the weak transferability in S-HAR. Inspired by this observation, we assume and empirically validate that smoothening the rugged loss landscape could potentially improve adversarial transferability in S-HAR. To this end, we propose the first \textbf{T}ransfer-based \textbf{A}ttack on \textbf{S}keletal \textbf{A}ction \textbf{R}ecognition, TASAR. TASAR explores the smoothed model posterior without requiring surrogate re-training, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike previous transfer-based attacks that treat each frame independently and overlook temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack gradient, effectively disrupting the spatial-temporal coherence of S-HARs. To exhaustively evaluate the effectiveness of existing methods and our method, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense methods. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the supplementary material.



## **20. POEX: Understanding and Mitigating Policy Executable Jailbreak Attacks against Embodied AI**

cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2412.16633v2) [paper-pdf](http://arxiv.org/pdf/2412.16633v2)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: Embodied AI systems are rapidly evolving due to the integration of LLMs as planning modules, which transform complex instructions into executable policies. However, LLMs are vulnerable to jailbreak attacks, which can generate malicious content. This paper investigates the feasibility and rationale behind applying traditional LLM jailbreak attacks to EAI systems. We aim to answer three questions: (1) Do traditional LLM jailbreak attacks apply to EAI systems? (2) What challenges arise if they do not? and (3) How can we defend against EAI jailbreak attacks? To this end, we first measure existing LLM-based EAI systems using a newly constructed dataset, i.e., the Harmful-RLbench. Our study confirms that traditional LLM jailbreak attacks are not directly applicable to EAI systems and identifies two unique challenges. First, the harmful text does not necessarily constitute harmful policies. Second, even if harmful policies can be generated, they are not necessarily executable by the EAI systems, which limits the potential risk. To facilitate a more comprehensive security analysis, we refine and introduce POEX, a novel red teaming framework that optimizes adversarial suffixes to induce harmful yet executable policies against EAI systems. The design of POEX employs adversarial constraints, policy evaluators, and suffix optimization to ensure successful policy execution while evading safety detection inside an EAI system. Experiments on the real-world robotic arm and simulator using Harmful-RLbench demonstrate the efficacy, highlighting severe safety vulnerabilities and high transferability across models. Finally, we propose prompt-based and model-based defenses, achieving an 85% success rate in mitigating attacks and enhancing safety awareness in EAI systems. Our findings underscore the urgent need for robust security measures to ensure the safe deployment of EAI in critical applications.



## **21. Detecting Backdoor Samples in Contrastive Language Image Pretraining**

cs.LG

ICLR2025

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.01385v2) [paper-pdf](http://arxiv.org/pdf/2502.01385v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: Contrastive language-image pretraining (CLIP) has been found to be vulnerable to poisoning backdoor attacks where the adversary can achieve an almost perfect attack success rate on CLIP models by poisoning only 0.01\% of the training dataset. This raises security concerns on the current practice of pretraining large-scale models on unscrutinized web data using CLIP. In this work, we analyze the representations of backdoor-poisoned samples learned by CLIP models and find that they exhibit unique characteristics in their local subspace, i.e., their local neighborhoods are far more sparse than that of clean samples. Based on this finding, we conduct a systematic study on detecting CLIP backdoor attacks and show that these attacks can be easily and efficiently detected by traditional density ratio-based local outlier detectors, whereas existing backdoor sample detection methods fail. Our experiments also reveal that an unintentional backdoor already exists in the original CC3M dataset and has been trained into a popular open-source model released by OpenCLIP. Based on our detector, one can clean up a million-scale web dataset (e.g., CC3M) efficiently within 15 minutes using 4 Nvidia A100 GPUs. The code is publicly available in our \href{https://github.com/HanxunH/Detect-CLIP-Backdoor-Samples}{GitHub repository}.



## **22. Confidence Elicitation: A New Attack Vector for Large Language Models**

cs.LG

Published in ICLR 2025. The code is publicly available at  https://github.com/Aniloid2/Confidence_Elicitation_Attacks

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2502.04643v2) [paper-pdf](http://arxiv.org/pdf/2502.04643v2)

**Authors**: Brian Formento, Chuan Sheng Foo, See-Kiong Ng

**Abstract**: A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large language models (LLMs) with billions of parameters suffer from adversarial attacks just like their earlier, smaller counterparts. However, the threat models have changed. Previously, having gray-box access, where input embeddings or output logits/probabilities were visible to the user, might have been reasonable. However, with the introduction of closed-source models, no information about the model is available apart from the generated output. This means that current black-box attacks can only utilize the final prediction to detect if an attack is successful. In this work, we investigate and demonstrate the potential of attack guidance, akin to using output probabilities, while having only black-box access in a classification setting. This is achieved through the ability to elicit confidence from the model. We empirically show that the elicited confidence is calibrated and not hallucinated for current LLMs. By minimizing the elicited confidence, we can therefore increase the likelihood of misclassification. Our new proposed paradigm demonstrates promising state-of-the-art results on three datasets across two models (LLaMA-3-8B-Instruct and Mistral-7B-Instruct-V0.3) when comparing our technique to existing hard-label black-box attack methods that introduce word-level substitutions.



## **23. ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time**

cs.CV

29pages

**SubmitDate**: 2025-02-10    [abs](http://arxiv.org/abs/2410.06625v2) [paper-pdf](http://arxiv.org/pdf/2410.06625v2)

**Authors**: Yi Ding, Bolian Li, Ruqi Zhang

**Abstract**: Vision Language Models (VLMs) have become essential backbones for multimodal intelligence, yet significant safety challenges limit their real-world application. While textual inputs are often effectively safeguarded, adversarial visual inputs can easily bypass VLM defense mechanisms. Existing defense methods are either resource-intensive, requiring substantial data and compute, or fail to simultaneously ensure safety and usefulness in responses. To address these limitations, we propose a novel two-phase inference-time alignment framework, Evaluating Then Aligning (ETA): 1) Evaluating input visual contents and output responses to establish a robust safety awareness in multimodal settings, and 2) Aligning unsafe behaviors at both shallow and deep levels by conditioning the VLMs' generative distribution with an interference prefix and performing sentence-level best-of-N to search the most harmless and helpful generation paths. Extensive experiments show that ETA outperforms baseline methods in terms of harmlessness, helpfulness, and efficiency, reducing the unsafe rate by 87.5% in cross-modality attacks and achieving 96.6% win-ties in GPT-4 helpfulness evaluation. The code is publicly available at https://github.com/DripNowhy/ETA.



## **24. A Conditional Tabular GAN-Enhanced Intrusion Detection System for Rare Attacks in IoT Networks**

cs.CR

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.06031v1) [paper-pdf](http://arxiv.org/pdf/2502.06031v1)

**Authors**: Safaa Menssouri, El Mehdi Amhoud

**Abstract**: Internet of things (IoT) networks, boosted by 6G technology, are transforming various industries. However, their widespread adoption introduces significant security risks, particularly in detecting rare but potentially damaging cyber-attacks. This makes the development of robust IDS crucial for monitoring network traffic and ensuring their safety. Traditional IDS often struggle with detecting rare attacks due to severe class imbalances in IoT data. In this paper, we propose a novel two-stage system called conditional tabular generative synthetic minority data generation with deep neural network (CTGSM-DNN). In the first stage, a conditional tabular generative adversarial network (CTGAN) is employed to generate synthetic data for rare attack classes. In the second stage, the SMOTEENN method is applied to improve dataset quality. The full study was conducted using the CSE-CIC-IDS2018 dataset, and we assessed the performance of the proposed IDS using different evaluation metrics. The experimental results demonstrated the effectiveness of the proposed multiclass classifier, achieving an overall accuracy of 99.90% and 80% accuracy in detecting rare attacks.



## **25. Detection of Physiological Data Tampering Attacks with Quantum Machine Learning**

quant-ph

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05966v1) [paper-pdf](http://arxiv.org/pdf/2502.05966v1)

**Authors**: Md. Saif Hassan Onim, Himanshu Thapliyal

**Abstract**: The widespread use of cloud-based medical devices and wearable sensors has made physiological data susceptible to tampering. These attacks can compromise the reliability of healthcare systems which can be critical and life-threatening. Detection of such data tampering is of immediate need. Machine learning has been used to detect anomalies in datasets but the performance of Quantum Machine Learning (QML) is still yet to be evaluated for physiological sensor data. Thus, our study compares the effectiveness of QML for detecting physiological data tampering, focusing on two types of white-box attacks: data poisoning and adversarial perturbation. The results show that QML models are better at identifying label-flipping attacks, achieving accuracy rates of 75%-95% depending on the data and attack severity. This superior performance is due to the ability of quantum algorithms to handle complex and high-dimensional data. However, both QML and classical models struggle to detect more sophisticated adversarial perturbation attacks, which subtly alter data without changing its statistical properties. Although QML performed poorly against this attack with around 45%-65% accuracy, it still outperformed classical algorithms in some cases.



## **26. A Practical Examination of AI-Generated Text Detectors for Large Language Models**

cs.CL

9 pages

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2412.05139v4) [paper-pdf](http://arxiv.org/pdf/2412.05139v4)

**Authors**: Brian Tufts, Xuandong Zhao, Lei Li

**Abstract**: The proliferation of large language models has raised growing concerns about their misuse, particularly in cases where AI-generated text is falsely attributed to human authors. Machine-generated content detectors claim to effectively identify such text under various conditions and from any language model. This paper critically evaluates these claims by assessing several popular detectors (RADAR, Wild, T5Sentinel, Fast-DetectGPT, PHD, LogRank, Binoculars) on a range of domains, datasets, and models that these detectors have not previously encountered. We employ various prompting strategies to simulate practical adversarial attacks, demonstrating that even moderate efforts can significantly evade detection. We emphasize the importance of the true positive rate at a specific false positive rate (TPR@FPR) metric and demonstrate that these detectors perform poorly in certain settings, with TPR@.01 as low as 0%. Our findings suggest that both trained and zero-shot detectors struggle to maintain high sensitivity while achieving a reasonable true positive rate.



## **27. Optimization under Attack: Resilience, Vulnerability, and the Path to Collapse**

cs.MA

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05954v1) [paper-pdf](http://arxiv.org/pdf/2502.05954v1)

**Authors**: Amal Aldawsari, Evangelos Pournaras

**Abstract**: Optimization is instrumental for improving operations of large-scale socio-technical infrastructures of Smart Cities, for instance, energy and traffic systems. In particular, understanding the performance of multi-agent discrete-choice combinatorial optimization under distributed adversary attacks is a compelling and underexplored problem, since multi-agent systems exhibit a large number of remote control variables that can influence in an unprecedented way the cost-effectiveness of distributed optimization heuristics. This paper unravels for the first time the trajectories of distributed optimization from resilience to vulnerability, and finally to collapse under varying adversary influence. Using real-world data to emulate over 28 billion multi-agent optimization scenarios, we exhaustively assess how the number of agents with different adversarial severity and network positioning influences optimization performance, including the influence on Pareto optimal points. With this novel large-scale dataset, made openly available as a benchmark, we disentangle how optimization remains resilient to adversaries and which adversary conditions are required to make optimization vulnerable or collapsed. These new findings can provide new insights for designing self-healing strategies for fault-tolerance and fault-correction in adversarial distributed optimization that have been missing so far.



## **28. Protecting Intellectual Property of EEG-based Neural Networks with Watermarking**

cs.LG

21 pages, 13 figures, and 6 tables

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05931v1) [paper-pdf](http://arxiv.org/pdf/2502.05931v1)

**Authors**: Ahmed Abdelaziz, Ahmed Fathi, Ahmed Fares

**Abstract**: EEG-based neural networks, pivotal in medical diagnosis and brain-computer interfaces, face significant intellectual property (IP) risks due to their reliance on sensitive neurophysiological data and resource-intensive development. Current watermarking methods, particularly those using abstract trigger sets, lack robust authentication and fail to address the unique challenges of EEG models. This paper introduces a cryptographic wonder filter-based watermarking framework tailored for EEG-based neural networks. Leveraging collision-resistant hashing and public-key encryption, the wonder filter embeds the watermark during training, ensuring minimal distortion ($\leq 5\%$ drop in EEG task accuracy) and high reliability (100\% watermark detection). The framework is rigorously evaluated against adversarial attacks, including fine-tuning, transfer learning, and neuron pruning. Results demonstrate persistent watermark retention, with classification accuracy for watermarked states remaining above 90\% even after aggressive pruning, while primary task performance degrades faster, deterring removal attempts. Piracy resistance is validated by the inability to embed secondary watermarks without severe accuracy loss ( $>10\%$ in EEGNet and CCNN models). Cryptographic hashing ensures authentication, reducing brute-force attack success probabilities. Evaluated on the DEAP dataset across models (CCNN, EEGNet, TSception), the method achieves $>99.4\%$ null-embedding accuracy, effectively eliminating false positives. By integrating wonder filters with EEG-specific adaptations, this work bridges a critical gap in IP protection for neurophysiological models, offering a secure, tamper-proof solution for healthcare and biometric applications. The framework's robustness against adversarial modifications underscores its potential to safeguard sensitive EEG models while maintaining diagnostic utility.



## **29. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

cs.LG

ICLR 2025

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2408.00315v2) [paper-pdf](http://arxiv.org/pdf/2408.00315v2)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.



## **30. Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails**

cs.CV

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05772v1) [paper-pdf](http://arxiv.org/pdf/2502.05772v1)

**Authors**: Yijun Yang, Lichao Wang, Xiao Yang, Lanqing Hong, Jun Zhu

**Abstract**: Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of generating unsafe responses. In response, leading companies have implemented Multi-Layered safety defenses, including alignment training, safety system prompts, and content moderation. However, their effectiveness against sophisticated adversarial attacks remains largely unexplored. In this paper, we propose MultiFaceted Attack, a novel attack framework designed to systematically bypass Multi-Layered Defenses in VLLMs. It comprises three complementary attack facets: Visual Attack that exploits the multimodal nature of VLLMs to inject toxic system prompts through images; Alignment Breaking Attack that manipulates the model's alignment mechanism to prioritize the generation of contrasting responses; and Adversarial Signature that deceives content moderators by strategically placing misleading information at the end of the response. Extensive evaluations on eight commercial VLLMs in a black-box setting demonstrate that MultiFaceted Attack achieves a 61.56% attack success rate, surpassing state-of-the-art methods by at least 42.18%.



## **31. Filter, Obstruct and Dilute: Defending Against Backdoor Attacks on Semi-Supervised Learning**

cs.LG

**SubmitDate**: 2025-02-09    [abs](http://arxiv.org/abs/2502.05755v1) [paper-pdf](http://arxiv.org/pdf/2502.05755v1)

**Authors**: Xinrui Wang, Chuanxing Geng, Wenhai Wan, Shao-yuan Li, Songcan Chen

**Abstract**: Recent studies have verified that semi-supervised learning (SSL) is vulnerable to data poisoning backdoor attacks. Even a tiny fraction of contaminated training data is sufficient for adversaries to manipulate up to 90\% of the test outputs in existing SSL methods. Given the emerging threat of backdoor attacks designed for SSL, this work aims to protect SSL against such risks, marking it as one of the few known efforts in this area. Specifically, we begin by identifying that the spurious correlations between the backdoor triggers and the target class implanted by adversaries are the primary cause of manipulated model predictions during the test phase. To disrupt these correlations, we utilize three key techniques: Gaussian Filter, complementary learning and trigger mix-up, which collectively filter, obstruct and dilute the influence of backdoor attacks in both data pre-processing and feature learning. Experimental results demonstrate that our proposed method, Backdoor Invalidator (BI), significantly reduces the average attack success rate from 84.7\% to 1.8\% across different state-of-the-art backdoor attacks. It is also worth mentioning that BI does not sacrifice accuracy on clean data and is supported by a theoretical guarantee of its generalization capability.



## **32. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

cs.CV

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05673v1) [paper-pdf](http://arxiv.org/pdf/2502.05673v1)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.



## **33. Rigid Body Adversarial Attacks**

cs.CV

17 pages, 14 figures, 3DV 2025

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05669v1) [paper-pdf](http://arxiv.org/pdf/2502.05669v1)

**Authors**: Aravind Ramakrishnan, David I. W. Levin, Alec Jacobson

**Abstract**: Due to their performance and simplicity, rigid body simulators are often used in applications where the objects of interest can considered very stiff. However, no material has infinite stiffness, which means there are potentially cases where the non-zero compliance of the seemingly rigid object can cause a significant difference between its trajectories when simulated in a rigid body or deformable simulator.   Similarly to how adversarial attacks are developed against image classifiers, we propose an adversarial attack against rigid body simulators. In this adversarial attack, we solve an optimization problem to construct perceptually rigid adversarial objects that have the same collision geometry and moments of mass to a reference object, so that they behave identically in rigid body simulations but maximally different in more accurate deformable simulations. We demonstrate the validity of our method by comparing simulations of several examples in commercially available simulators.



## **34. Adversarial Machine Learning: Attacks, Defenses, and Open Challenges**

cs.CR

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05637v1) [paper-pdf](http://arxiv.org/pdf/2502.05637v1)

**Authors**: Pranav K Jha

**Abstract**: Adversarial Machine Learning (AML) addresses vulnerabilities in AI systems where adversaries manipulate inputs or training data to degrade performance. This article provides a comprehensive analysis of evasion and poisoning attacks, formalizes defense mechanisms with mathematical rigor, and discusses the challenges of implementing robust solutions in adaptive threat models. Additionally, it highlights open challenges in certified robustness, scalability, and real-world deployment.



## **35. Democratic Training Against Universal Adversarial Perturbations**

cs.LG

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05542v1) [paper-pdf](http://arxiv.org/pdf/2502.05542v1)

**Authors**: Bing Sun, Jun Sun, Wei Zhao

**Abstract**: Despite their advances and success, real-world deep neural networks are known to be vulnerable to adversarial attacks. Universal adversarial perturbation, an input-agnostic attack, poses a serious threat for them to be deployed in security-sensitive systems. In this case, a single universal adversarial perturbation deceives the model on a range of clean inputs without requiring input-specific optimization, which makes it particularly threatening. In this work, we observe that universal adversarial perturbations usually lead to abnormal entropy spectrum in hidden layers, which suggests that the prediction is dominated by a small number of ``feature'' in such cases (rather than democratically by many features). Inspired by this, we propose an efficient yet effective defense method for mitigating UAPs called \emph{Democratic Training} by performing entropy-based model enhancement to suppress the effect of the universal adversarial perturbations in a given model. \emph{Democratic Training} is evaluated with 7 neural networks trained on 5 benchmark datasets and 5 types of state-of-the-art universal adversarial attack methods. The results show that it effectively reduces the attack success rate, improves model robustness and preserves the model accuracy on clean samples.



## **36. Do Spikes Protect Privacy? Investigating Black-Box Model Inversion Attacks in Spiking Neural Networks**

cs.LG

7 pages, 4 figures

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05509v1) [paper-pdf](http://arxiv.org/pdf/2502.05509v1)

**Authors**: Hamed Poursiami, Ayana Moshruba, Maryam Parsa

**Abstract**: As machine learning models become integral to security-sensitive applications, concerns over data leakage from adversarial attacks continue to rise. Model Inversion (MI) attacks pose a significant privacy threat by enabling adversaries to reconstruct training data from model outputs. While MI attacks on Artificial Neural Networks (ANNs) have been widely studied, Spiking Neural Networks (SNNs) remain largely unexplored in this context. Due to their event-driven and discrete computations, SNNs introduce fundamental differences in information processing that may offer inherent resistance to such attacks. A critical yet underexplored aspect of this threat lies in black-box settings, where attackers operate through queries without direct access to model parameters or gradients-representing a more realistic adversarial scenario in deployed systems. This work presents the first study of black-box MI attacks on SNNs. We adapt a generative adversarial MI framework to the spiking domain by incorporating rate-based encoding for input transformation and decoding mechanisms for output interpretation. Our results show that SNNs exhibit significantly greater resistance to MI attacks than ANNs, as demonstrated by degraded reconstructions, increased instability in attack convergence, and overall reduced attack effectiveness across multiple evaluation metrics. Further analysis suggests that the discrete and temporally distributed nature of SNN decision boundaries disrupts surrogate modeling, limiting the attacker's ability to approximate the target model.



## **37. Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey**

cs.CL

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.06872v1) [paper-pdf](http://arxiv.org/pdf/2502.06872v1)

**Authors**: Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Yinglong Xia, Krishnaram Kenthapadi, Ryan Rossi, Franck Dernoncourt, Md Mehrab Tanjim, Nesreen Ahmed, Xiaorui Liu, Wenqi Fan, Erik Blasch, Yu Wang, Meng Jiang, Tyler Derr

**Abstract**: Retrieval-Augmented Generation (RAG) is an advanced technique designed to address the challenges of Artificial Intelligence-Generated Content (AIGC). By integrating context retrieval into content generation, RAG provides reliable and up-to-date external knowledge, reduces hallucinations, and ensures relevant context across a wide range of tasks. However, despite RAG's success and potential, recent studies have shown that the RAG paradigm also introduces new risks, including robustness issues, privacy concerns, adversarial attacks, and accountability issues. Addressing these risks is critical for future applications of RAG systems, as they directly impact their trustworthiness. Although various methods have been developed to improve the trustworthiness of RAG methods, there is a lack of a unified perspective and framework for research in this topic. Thus, in this paper, we aim to address this gap by providing a comprehensive roadmap for developing trustworthy RAG systems. We place our discussion around five key perspectives: reliability, privacy, safety, fairness, explainability, and accountability. For each perspective, we present a general framework and taxonomy, offering a structured approach to understanding the current challenges, evaluating existing solutions, and identifying promising future research directions. To encourage broader adoption and innovation, we also highlight the downstream applications where trustworthy RAG systems have a significant impact.



## **38. SMaCk: Efficient Instruction Cache Attacks via Self-Modifying Code Conflicts**

cs.CR

Proceedings of the 30th ACM International Conference on Architectural  Support for Programming Languages and Operating Systems (ASPLOS) accepted

**SubmitDate**: 2025-02-08    [abs](http://arxiv.org/abs/2502.05429v1) [paper-pdf](http://arxiv.org/pdf/2502.05429v1)

**Authors**: Seonghun Son, Daniel Moghimi, Berk Gulmezoglu

**Abstract**: Self-modifying code (SMC) allows programs to alter their own instructions, optimizing performance and functionality on x86 processors. Despite its benefits, SMC introduces unique microarchitectural behaviors that can be exploited for malicious purposes. In this paper, we explore the security implications of SMC by examining how specific x86 instructions affecting instruction cache lines lead to measurable timing discrepancies between cache hits and misses. These discrepancies facilitate refined cache attacks, making them less noisy and more effective. We introduce novel attack techniques that leverage these timing variations to enhance existing methods such as Prime+Probe and Flush+Reload. Our advanced techniques allow adversaries to more precisely attack cryptographic keys and create covert channels akin to Spectre across various x86 platforms. Finally, we propose a dynamic detection methodology utilizing hardware performance counters to mitigate these enhanced threats.



## **39. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05374v1) [paper-pdf](http://arxiv.org/pdf/2502.05374v1)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.



## **40. Neural Encrypted State Transduction for Ransomware Classification: A Novel Approach Using Cryptographic Flow Residuals**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05341v1) [paper-pdf](http://arxiv.org/pdf/2502.05341v1)

**Authors**: Barnaby Fortescue, Edmund Hawksmoor, Alistair Wetherington, Frederick Marlowe, Kevin Pekepok

**Abstract**: Encrypted behavioral patterns provide a unique avenue for classifying complex digital threats without reliance on explicit feature extraction, enabling detection frameworks to remain effective even when conventional static and behavioral methodologies fail. A novel approach based on Neural Encrypted State Transduction (NEST) is introduced to analyze cryptographic flow residuals and classify threats through their encrypted state transitions, mitigating evasion tactics employed through polymorphic and obfuscated attack strategies. The mathematical formulation of NEST leverages transduction principles to map state transitions dynamically, enabling high-confidence classification without requiring direct access to decrypted execution traces. Experimental evaluations demonstrate that the proposed framework achieves improved detection accuracy across multiple ransomware families while exhibiting resilience against adversarial perturbations and previously unseen attack variants. The model maintains competitive processing efficiency, offering a practical balance between classification performance and computational resource constraints, making it suitable for large-scale security deployments. Comparative assessments reveal that NEST consistently outperforms baseline classification models, particularly in detecting ransomware samples employing delayed encryption, entropy-based obfuscation, and memory-resident execution techniques. The capacity to generalize across diverse execution environments reinforces the applicability of encrypted transduction methodologies in adversarial classification tasks beyond conventional malware detection pipelines. The integration of residual learning mechanisms within the transduction layers further enhances classification robustness, minimizing both false positives and misclassification rates across varied operational contexts.



## **41. ADAPT to Robustify Prompt Tuning Vision Transformers**

cs.LG

Published in Transactions on Machine Learning Research (2025)

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2403.13196v2) [paper-pdf](http://arxiv.org/pdf/2403.13196v2)

**Authors**: Masih Eskandar, Tooba Imtiaz, Zifeng Wang, Jennifer Dy

**Abstract**: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our method achieves competitive robust accuracy of ~40% w.r.t. SOTA robustness methods using full-model fine-tuning, by tuning only ~1% of the number of parameters.



## **42. Do Unlearning Methods Remove Information from Language Model Weights?**

cs.LG

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2410.08827v3) [paper-pdf](http://arxiv.org/pdf/2410.08827v3)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods for information learned during pretraining, revealing the limitations of these methods in removing information from the model weights. Our results also suggest that unlearning evaluations that measure unlearning robustness on information learned during an additional fine-tuning phase may overestimate robustness compared to evaluations that attempt to unlearn information learned during pretraining.



## **43. Federated Learning for Anomaly Detection in Energy Consumption Data: Assessing the Vulnerability to Adversarial Attacks**

cs.LG

12th IEEE Conference on Technologies for Sustainability

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05041v1) [paper-pdf](http://arxiv.org/pdf/2502.05041v1)

**Authors**: Yohannis Kifle Telila, Damitha Senevirathne, Dumindu Tissera, Apurva Narayan, Miriam A. M. Capretz, Katarina Grolinger

**Abstract**: Anomaly detection is crucial in the energy sector to identify irregular patterns indicating equipment failures, energy theft, or other issues. Machine learning techniques for anomaly detection have achieved great success, but are typically centralized, involving sharing local data with a central server which raises privacy and security concerns. Federated Learning (FL) has been gaining popularity as it enables distributed learning without sharing local data. However, FL depends on neural networks, which are vulnerable to adversarial attacks that manipulate data, leading models to make erroneous predictions. While adversarial attacks have been explored in the image domain, they remain largely unexplored in time series problems, especially in the energy domain. Moreover, the effect of adversarial attacks in the FL setting is also mostly unknown. This paper assesses the vulnerability of FL-based anomaly detection in energy data to adversarial attacks. Specifically, two state-of-the-art models, Long Short Term Memory (LSTM) and Transformers, are used to detect anomalies in an FL setting, and two white-box attack methods, Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), are employed to perturb the data. The results show that FL is more sensitive to PGD attacks than to FGSM attacks, attributed to PGD's iterative nature, resulting in an accuracy drop of over 10% even with naive, weaker attacks. Moreover, FL is more affected by these attacks than centralized learning, highlighting the need for defense mechanisms in FL.



## **44. Robust Graph Learning Against Adversarial Evasion Attacks via Prior-Free Diffusion-Based Structure Purification**

cs.LG

Accepted for poster at WWW 2025

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.05000v1) [paper-pdf](http://arxiv.org/pdf/2502.05000v1)

**Authors**: Jiayi Luo, Qingyun Sun, Haonan Yuan, Xingcheng Fu, Jianxin Li

**Abstract**: Adversarial evasion attacks pose significant threats to graph learning, with lines of studies that have improved the robustness of Graph Neural Networks (GNNs). However, existing works rely on priors about clean graphs or attacking strategies, which are often heuristic and inconsistent. To achieve robust graph learning over different types of evasion attacks and diverse datasets, we investigate this problem from a prior-free structure purification perspective. Specifically, we propose a novel Diffusion-based Structure Purification framework named DiffSP, which creatively incorporates the graph diffusion model to learn intrinsic distributions of clean graphs and purify the perturbed structures by removing adversaries under the direction of the captured predictive patterns without relying on priors. DiffSP is divided into the forward diffusion process and the reverse denoising process, during which structure purification is achieved. To avoid valuable information loss during the forward process, we propose an LID-driven nonisotropic diffusion mechanism to selectively inject noise anisotropically. To promote semantic alignment between the clean graph and the purified graph generated during the reverse process, we reduce the generation uncertainty by the proposed graph transfer entropy guided denoising mechanism. Extensive experiments demonstrate the superior robustness of DiffSP against evasion attacks.



## **45. Securing 5G Bootstrapping: A Two-Layer IBS Authentication Protocol**

cs.CR

13 pages, 4 figures, 3 tables. This work has been submitted to the  IEEE for possible publication

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04915v1) [paper-pdf](http://arxiv.org/pdf/2502.04915v1)

**Authors**: Yilu Dong, Rouzbeh Behnia, Attila A. Yavuz, Syed Rafiul Hussain

**Abstract**: The lack of authentication during the initial bootstrapping phase between cellular devices and base stations allows attackers to deploy fake base stations and send malicious messages to the devices. These attacks have been a long-existing problem in cellular networks, enabling adversaries to launch denial-of-service (DoS), information leakage, and location-tracking attacks. While some defense mechanisms are introduced in 5G, (e.g., encrypting user identifiers to mitigate IMSI catchers), the initial communication between devices and base stations remains unauthenticated, leaving a critical security gap. To address this, we propose E2IBS, a novel and efficient two-layer identity-based signature scheme designed for seamless integration with existing cellular protocols. We implement E2IBS on an open-source 5G stack and conduct a comprehensive performance evaluation against alternative solutions. Compared to the state-of-the-art Schnorr-HIBS, E2IBS reduces attack surfaces, enables fine-grained lawful interception, and achieves 2x speed in verification, making it a practical solution for securing 5G base station authentication.



## **46. From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection**

cs.CR

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2412.10198v2) [paper-pdf](http://arxiv.org/pdf/2412.10198v2)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.



## **47. DMPA: Model Poisoning Attacks on Decentralized Federated Learning for Model Differences**

cs.LG

8 pages, 3 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04771v1) [paper-pdf](http://arxiv.org/pdf/2502.04771v1)

**Authors**: Chao Feng, Yunlong Li, Yuanzhe Gao, Alberto Huertas Celdrán, Jan von der Assen, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated learning (FL) has garnered significant attention as a prominent privacy-preserving Machine Learning (ML) paradigm. Decentralized FL (DFL) eschews traditional FL's centralized server architecture, enhancing the system's robustness and scalability. However, these advantages of DFL also create new vulnerabilities for malicious participants to execute adversarial attacks, especially model poisoning attacks. In model poisoning attacks, malicious participants aim to diminish the performance of benign models by creating and disseminating the compromised model. Existing research on model poisoning attacks has predominantly concentrated on undermining global models within the Centralized FL (CFL) paradigm, while there needs to be more research in DFL. To fill the research gap, this paper proposes an innovative model poisoning attack called DMPA. This attack calculates the differential characteristics of multiple malicious client models and obtains the most effective poisoning strategy, thereby orchestrating a collusive attack by multiple participants. The effectiveness of this attack is validated across multiple datasets, with results indicating that the DMPA approach consistently surpasses existing state-of-the-art FL model poisoning attack strategies.



## **48. Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage**

cs.LG

There is something wrong with the order of Figures 8-11. And I need  to add an experiment with differential privacy quantization mutual  information value

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.02913v3) [paper-pdf](http://arxiv.org/pdf/2502.02913v3)

**Authors**: Jiayang Meng, Tao Huang, Hong Chen, Xin Shi, Qingyu Huang, Chen Hou

**Abstract**: The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stemming from gradient leakage during training. Current privacy assessments primarily rely on post-training attack simulations. However, these methods are inherently reactive, unable to encompass all potential attack scenarios, and often based on idealized adversarial assumptions. These limitations underscore the need for proactive approaches to privacy risk assessment during the training process. To address this gap, we propose the concept of privacy tokens, which are derived directly from private gradients during training. Privacy tokens encapsulate gradient features and, when combined with data features, offer valuable insights into the extent of private information leakage from training data, enabling real-time measurement of privacy risks without relying on adversarial attack simulations. Additionally, we employ Mutual Information (MI) as a robust metric to quantify the relationship between training data and gradients, providing precise and continuous assessments of privacy leakage throughout the training process. Extensive experiments validate our framework, demonstrating the effectiveness of privacy tokens and MI in identifying and quantifying privacy risks. This proactive approach marks a significant advancement in privacy monitoring, promoting the safer deployment of deep learning models in sensitive applications.



## **49. Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2025-02-07    [abs](http://arxiv.org/abs/2502.04679v1) [paper-pdf](http://arxiv.org/pdf/2502.04679v1)

**Authors**: Chashi Mahiul Islam, Samuel Jacob Chacko, Mao Nishino, Xiuwen Liu

**Abstract**: While transformer-based models dominate NLP and vision applications, their underlying mechanisms to map the input space to the label space semantically are not well understood. In this paper, we study the sources of known representation vulnerabilities of vision transformers (ViT), where perceptually identical images can have very different representations and semantically unrelated images can have the same representation. Our analysis indicates that imperceptible changes to the input can result in significant representation changes, particularly in later layers, suggesting potential instabilities in the performance of ViTs. Our comprehensive study reveals that adversarial effects, while subtle in early layers, propagate and amplify through the network, becoming most pronounced in middle to late layers. This insight motivates the development of NeuroShield-ViT, a novel defense mechanism that strategically neutralizes vulnerable neurons in earlier layers to prevent the cascade of adversarial effects. We demonstrate NeuroShield-ViT's effectiveness across various attacks, particularly excelling against strong iterative attacks, and showcase its remarkable zero-shot generalization capabilities. Without fine-tuning, our method achieves a competitive accuracy of 77.8% on adversarial examples, surpassing conventional robustness methods. Our results shed new light on how adversarial effects propagate through ViT layers, while providing a promising approach to enhance the robustness of vision transformers against adversarial attacks. Additionally, they provide a promising approach to enhance the robustness of vision transformers against adversarial attacks.



## **50. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2025-02-06    [abs](http://arxiv.org/abs/2410.10572v2) [paper-pdf](http://arxiv.org/pdf/2410.10572v2)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



