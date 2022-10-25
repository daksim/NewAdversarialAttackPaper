# Latest Adversarial Attack Papers
**update at 2022-10-26 06:31:39**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Probabilistic Categorical Adversarial Attack & Adversarial Training**

cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.09364v2) [paper-pdf](http://arxiv.org/pdf/2210.09364v2)

**Authors**: Pengfei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.



## **2. Driver Locations Harvesting Attack on pRide**

cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13263v1) [paper-pdf](http://arxiv.org/pdf/2210.13263v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.



## **3. SealClub: Computer-aided Paper Document Authentication**

cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.07884v2) [paper-pdf](http://arxiv.org/pdf/2210.07884v2)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.



## **4. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

cs.CV

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.08189v3) [paper-pdf](http://arxiv.org/pdf/2204.08189v3)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.



## **5. Ares: A System-Oriented Wargame Framework for Adversarial ML**

cs.LG

Presented at the DLS Workshop at S&P 2022

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.12952v1) [paper-pdf](http://arxiv.org/pdf/2210.12952v1)

**Authors**: Farhan Ahmed, Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Since the discovery of adversarial attacks against machine learning models nearly a decade ago, research on adversarial machine learning has rapidly evolved into an eternal war between defenders, who seek to increase the robustness of ML models against adversarial attacks, and adversaries, who seek to develop better attacks capable of weakening or defeating these defenses. This domain, however, has found little buy-in from ML practitioners, who are neither overtly concerned about these attacks affecting their systems in the real world nor are willing to trade off the accuracy of their models in pursuit of robustness against these attacks.   In this paper, we motivate the design and implementation of Ares, an evaluation framework for adversarial ML that allows researchers to explore attacks and defenses in a realistic wargame-like environment. Ares frames the conflict between the attacker and defender as two agents in a reinforcement learning environment with opposing objectives. This allows the introduction of system-level evaluation metrics such as time to failure and evaluation of complex strategies such as moving target defenses. We provide the results of our initial exploration involving a white-box attacker against an adversarially trained defender.



## **6. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

cs.LG

Accepted to EMNLP 2022, 9 pages and Appendix

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.14017v2) [paper-pdf](http://arxiv.org/pdf/2204.14017v2)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstract**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through rare word embeddings of NLP models. In text classification, less than 1% of adversary clients suffices to manipulate the model output without any drop in the performance on clean sentences. For a less complex dataset, a mere 0.1% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called Gradient Ensemble, which enhances the backdoor performance in all our experimental settings.



## **7. TextHacker: Learning based Hybrid Local Search Algorithm for Text Hard-label Adversarial Attack**

cs.CL

Accepted by EMNLP 2022 Findings, Code is available at  https://github.com/JHL-HUST/TextHacker

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2201.08193v2) [paper-pdf](http://arxiv.org/pdf/2201.08193v2)

**Authors**: Zhen Yu, Xiaosen Wang, Wanxiang Che, Kun He

**Abstract**: Existing textual adversarial attacks usually utilize the gradient or prediction confidence to generate adversarial examples, making it hard to be deployed in real-world applications. To this end, we consider a rarely investigated but more rigorous setting, namely hard-label attack, in which the attacker can only access the prediction label. In particular, we find we can learn the importance of different words via the change on prediction label caused by word substitutions on the adversarial examples. Based on this observation, we propose a novel adversarial attack, termed Text Hard-label attacker (TextHacker). TextHacker randomly perturbs lots of words to craft an adversarial example. Then, TextHacker adopts a hybrid local search algorithm with the estimation of word importance from the attack history to minimize the adversarial perturbation. Extensive evaluations for text classification and textual entailment show that TextHacker significantly outperforms existing hard-label attacks regarding the attack performance as well as adversary quality.



## **8. A Secure Design Pattern Approach Toward Tackling Lateral-Injection Attacks**

cs.CR

4 pages, 3 figures. Accepted to The 15th IEEE International  Conference on Security of Information and Networks (SIN)

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12877v1) [paper-pdf](http://arxiv.org/pdf/2210.12877v1)

**Authors**: Chidera Biringa, Gökhan Kul

**Abstract**: Software weaknesses that create attack surfaces for adversarial exploits, such as lateral SQL injection (LSQLi) attacks, are usually introduced during the design phase of software development. Security design patterns are sometimes applied to tackle these weaknesses. However, due to the stealthy nature of lateral-based attacks, employing traditional security patterns to address these threats is insufficient. Hence, we present SEAL, a secure design that extrapolates architectural, design, and implementation abstraction levels to delegate security strategies toward tackling LSQLi attacks. We evaluated SEAL using case study software, where we assumed the role of an adversary and injected several attack vectors tasked with compromising the confidentiality and integrity of its database. Our evaluation of SEAL demonstrated its capacity to address LSQLi attacks.



## **9. TAPE: Assessing Few-shot Russian Language Understanding**

cs.CL

Accepted to EMNLP 2022 Findings

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12813v1) [paper-pdf](http://arxiv.org/pdf/2210.12813v1)

**Authors**: Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

**Abstract**: Recent advances in zero-shot and few-shot learning have shown promise for a scope of research and practical purposes. However, this fast-growing area lacks standardized evaluation suites for non-English languages, hindering progress outside the Anglo-centric paradigm. To address this line of research, we propose TAPE (Text Attack and Perturbation Evaluation), a novel benchmark that includes six more complex NLU tasks for Russian, covering multi-hop reasoning, ethical concepts, logic and commonsense knowledge. The TAPE's design focuses on systematic zero-shot and few-shot NLU evaluation: (i) linguistic-oriented adversarial attacks and perturbations for analyzing robustness, and (ii) subpopulations for nuanced interpretation. The detailed analysis of testing the autoregressive baselines indicates that simple spelling-based perturbations affect the performance the most, while paraphrasing the input has a more negligible effect. At the same time, the results demonstrate a significant gap between the neural and human baselines for most tasks. We publicly release TAPE (tape-benchmark.com) to foster research on robust LMs that can generalize to new tasks when little to no supervision is available.



## **10. GANI: Global Attacks on Graph Neural Networks via Imperceptible Node Injections**

cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12598v1) [paper-pdf](http://arxiv.org/pdf/2210.12598v1)

**Authors**: Junyuan Fang, Haixian Wen, Jiajing Wu, Qi Xuan, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks (GNNs) have found successful applications in various graph-related tasks. However, recent studies have shown that many GNNs are vulnerable to adversarial attacks. In a vast majority of existing studies, adversarial attacks on GNNs are launched via direct modification of the original graph such as adding/removing links, which may not be applicable in practice. In this paper, we focus on a realistic attack operation via injecting fake nodes. The proposed Global Attack strategy via Node Injection (GANI) is designed under the comprehensive consideration of an unnoticeable perturbation setting from both structure and feature domains. Specifically, to make the node injections as imperceptible and effective as possible, we propose a sampling operation to determine the degree of the newly injected nodes, and then generate features and select neighbors for these injected nodes based on the statistical information of features and evolutionary perturbations obtained from a genetic algorithm, respectively. In particular, the proposed feature generation mechanism is suitable for both binary and continuous node features. Extensive experimental results on benchmark datasets against both general and defended GNNs show strong attack performance of GANI. Moreover, the imperceptibility analyses also demonstrate that GANI achieves a relatively unnoticeable injection on benchmark datasets.



## **11. Efficient (Soft) Q-Learning for Text Generation with Limited Good Data**

cs.CL

Code available at  https://github.com/HanGuo97/soft-Q-learning-for-text-generation

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2106.07704v4) [paper-pdf](http://arxiv.org/pdf/2106.07704v4)

**Authors**: Han Guo, Bowen Tan, Zhengzhong Liu, Eric P. Xing, Zhiting Hu

**Abstract**: Maximum likelihood estimation (MLE) is the predominant algorithm for training text generation models. This paradigm relies on direct supervision examples, which is not applicable to many emerging applications, such as generating adversarial attacks or generating prompts to control language models. Reinforcement learning (RL) on the other hand offers a more flexible solution by allowing users to plug in arbitrary task metrics as reward. Yet previous RL algorithms for text generation, such as policy gradient (on-policy RL) and Q-learning (off-policy RL), are often notoriously inefficient or unstable to train due to the large sequence space and the sparse reward received only at the end of sequences. In this paper, we introduce a new RL formulation for text generation from the soft Q-learning (SQL) perspective. It enables us to draw from the latest RL advances, such as path consistency learning, to combine the best of on-/off-policy updates, and learn effectively from sparse reward. We apply the approach to a wide range of novel text generation tasks, including learning from noisy/negative examples, adversarial attacks, and prompt generation. Experiments show our approach consistently outperforms both task-specialized algorithms and the previous RL methods.



## **12. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2206.02829v3) [paper-pdf](http://arxiv.org/pdf/2206.02829v3)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstract**: Offline reinforcement learning (RL) provides a promising direction to exploit massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.



## **13. ADDMU: Detection of Far-Boundary Adversarial Examples with Data and Model Uncertainty Estimation**

cs.CL

18 pages, EMNLP 2022, main conference, long paper

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2210.12396v1) [paper-pdf](http://arxiv.org/pdf/2210.12396v1)

**Authors**: Fan Yin, Yao Li, Cho-Jui Hsieh, Kai-Wei Chang

**Abstract**: Adversarial Examples Detection (AED) is a crucial defense technique against adversarial attacks and has drawn increasing attention from the Natural Language Processing (NLP) community. Despite the surge of new AED methods, our studies show that existing methods heavily rely on a shortcut to achieve good performance. In other words, current search-based adversarial attacks in NLP stop once model predictions change, and thus most adversarial examples generated by those attacks are located near model decision boundaries. To surpass this shortcut and fairly evaluate AED methods, we propose to test AED methods with \textbf{F}ar \textbf{B}oundary (\textbf{FB}) adversarial examples. Existing methods show worse than random guess performance under this scenario. To overcome this limitation, we propose a new technique, \textbf{ADDMU}, \textbf{a}dversary \textbf{d}etection with \textbf{d}ata and \textbf{m}odel \textbf{u}ncertainty, which combines two types of uncertainty estimation for both regular and FB adversarial example detection. Our new method outperforms previous methods by 3.6 and 6.0 \emph{AUC} points under each scenario. Finally, our analysis shows that the two types of uncertainty provided by \textbf{ADDMU} can be leveraged to characterize adversarial examples and identify the ones that contribute most to model's robustness in adversarial training.



## **14. TCAB: A Large-Scale Text Classification Attack Benchmark**

cs.LG

32 pages, 7 figures, and 14 tables

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12233v1) [paper-pdf](http://arxiv.org/pdf/2210.12233v1)

**Authors**: Kalyani Asthana, Zhouhang Xie, Wencong You, Adam Noack, Jonathan Brophy, Sameer Singh, Daniel Lowd

**Abstract**: We introduce the Text Classification Attack Benchmark (TCAB), a dataset for analyzing, understanding, detecting, and labeling adversarial attacks against text classifiers. TCAB includes 1.5 million attack instances, generated by twelve adversarial attacks targeting three classifiers trained on six source datasets for sentiment analysis and abuse detection in English. Unlike standard text classification, text attacks must be understood in the context of the target classifier that is being attacked, and thus features of the target classifier are important as well. TCAB includes all attack instances that are successful in flipping the predicted label; a subset of the attacks are also labeled by human annotators to determine how frequently the primary semantics are preserved. The process of generating attacks is automated, so that TCAB can easily be extended to incorporate new text attacks and better classifiers as they are developed. In addition to the primary tasks of detecting and labeling attacks, TCAB can also be used for attack localization, attack target labeling, and attack characterization. TCAB code and dataset are available at https://react-nlp.github.io/tcab/.



## **15. The Dark Side of AutoML: Towards Architectural Backdoor Search**

cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12179v1) [paper-pdf](http://arxiv.org/pdf/2210.12179v1)

**Authors**: Ren Pang, Changjiang Li, Zhaohan Xi, Shouling Ji, Ting Wang

**Abstract**: This paper asks the intriguing question: is it possible to exploit neural architecture search (NAS) as a new attack vector to launch previously improbable attacks? Specifically, we present EVAS, a new attack that leverages NAS to find neural architectures with inherent backdoors and exploits such vulnerability using input-aware triggers. Compared with existing attacks, EVAS demonstrates many interesting properties: (i) it does not require polluting training data or perturbing model parameters; (ii) it is agnostic to downstream fine-tuning or even re-training from scratch; (iii) it naturally evades defenses that rely on inspecting model parameters or training data. With extensive evaluation on benchmark datasets, we show that EVAS features high evasiveness, transferability, and robustness, thereby expanding the adversary's design spectrum. We further characterize the mechanisms underlying EVAS, which are possibly explainable by architecture-level ``shortcuts'' that recognize trigger patterns. This work raises concerns about the current practice of NAS and points to potential directions to develop effective countermeasures.



## **16. Evolution of Neural Tangent Kernels under Benign and Adversarial Training**

cs.LG

Accepted to the Conference on Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12030v1) [paper-pdf](http://arxiv.org/pdf/2210.12030v1)

**Authors**: Noel Loo, Ramin Hasani, Alexander Amini, Daniela Rus

**Abstract**: Two key challenges facing modern deep learning are mitigating deep networks' vulnerability to adversarial attacks and understanding deep learning's generalization capabilities. Towards the first issue, many defense strategies have been developed, with the most common being Adversarial Training (AT). Towards the second challenge, one of the dominant theories that has emerged is the Neural Tangent Kernel (NTK) -- a characterization of neural network behavior in the infinite-width limit. In this limit, the kernel is frozen, and the underlying feature map is fixed. In finite widths, however, there is evidence that feature learning happens at the earlier stages of the training (kernel learning) before a second phase where the kernel remains fixed (lazy training). While prior work has aimed at studying adversarial vulnerability through the lens of the frozen infinite-width NTK, there is no work that studies the adversarial robustness of the empirical/finite NTK during training. In this work, we perform an empirical study of the evolution of the empirical NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the empirical NTK rapidly converges to a different kernel (and feature map) than standard training. This new kernel provides adversarial robustness, even when non-robust training is performed on top of it. Furthermore, we find that adversarial training on top of a fixed kernel can yield a classifier with $76.1\%$ robust accuracy under PGD attacks with $\varepsilon = 4/255$ on CIFAR-10.



## **17. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.03755v2) [paper-pdf](http://arxiv.org/pdf/2209.03755v2)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.



## **18. Assaying Out-Of-Distribution Generalization in Transfer Learning**

cs.LG

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2207.09239v2) [paper-pdf](http://arxiv.org/pdf/2207.09239v2)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstract**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.



## **19. A Survey of Machine Unlearning**

cs.LG

discuss new and recent works as well as proof-reading

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.02299v5) [paper-pdf](http://arxiv.org/pdf/2209.02299v5)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.



## **20. Identifying Human Strategies for Generating Word-Level Adversarial Examples**

cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11598v1) [paper-pdf](http://arxiv.org/pdf/2210.11598v1)

**Authors**: Maximilian Mozes, Bennett Kleinberg, Lewis D. Griffin

**Abstract**: Adversarial examples in NLP are receiving increasing research attention. One line of investigation is the generation of word-level adversarial examples against fine-tuned Transformer models that preserve naturalness and grammaticality. Previous work found that human- and machine-generated adversarial examples are comparable in their naturalness and grammatical correctness. Most notably, humans were able to generate adversarial examples much more effortlessly than automated attacks. In this paper, we provide a detailed analysis of exactly how humans create these adversarial examples. By exploring the behavioural patterns of human workers during the generation process, we identify statistically significant tendencies based on which words humans prefer to select for adversarial replacement (e.g., word frequencies, word saliencies, sentiment) as well as where and when words are replaced in an input sequence. With our findings, we seek to inspire efforts that harness human strategies for more robust NLP models.



## **21. Similarity of Neural Architectures Based on Input Gradient Transferability**

cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11407v1) [paper-pdf](http://arxiv.org/pdf/2210.11407v1)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In this paper, we aim to design a quantitative similarity function between two neural architectures. Specifically, we define a model similarity using input gradient transferability. We generate adversarial samples of two networks and measure the average accuracy of the networks on adversarial samples of each other. If two networks are highly correlated, then the attack transferability will be high, resulting in high similarity. Using the similarity score, we investigate two topics: (1) Which network component contributes to the model diversity? (2) How does model diversity affect practical scenarios? We answer the first question by providing feature importance analysis and clustering analysis. The second question is validated by two different scenarios: model ensemble and knowledge distillation. Our findings show that model diversity takes a key role when interacting with different neural architectures. For example, we found that more diversity leads to better ensemble performance. We also observe that the relationship between teacher and student networks and distillation performance depends on the choice of the base architecture of the teacher and student networks. We expect our analysis tool helps a high-level understanding of differences between various neural architectures as well as practical guidance when using multiple architectures.



## **22. Surprises in adversarially-trained linear regression**

stat.ML

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2205.12695v2) [paper-pdf](http://arxiv.org/pdf/2205.12695v2)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against such examples. It is formulated as a min-max problem, searching for the best solution when the training data was corrupted by the worst-case attacks. For linear regression problems, adversarial training can be formulated as a convex problem. We use this reformulation to make two technical contributions: First, we formulate the training problem as an instance of robust regression to reveal its connection to parameter-shrinking methods, specifically that $\ell_\infty$-adversarial training produces sparse solutions. Secondly, we study adversarial training in the overparameterized regime, i.e. when there are more parameters than data. We prove that adversarial training with small disturbances gives the solution with the minimum-norm that interpolates the training data. Ridge regression and lasso approximate such interpolating solutions as their regularization parameter vanishes. By contrast, for adversarial training, the transition into the interpolation regime is abrupt and for non-zero values of disturbance. This result is proved and illustrated with numerical examples.



## **23. Attacking Motion Estimation with Adversarial Snow**

cs.CV

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11242v1) [paper-pdf](http://arxiv.org/pdf/2210.11242v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks for motion estimation (optical flow) optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, we exploit a real-world weather phenomenon for a novel attack with adversarially optimized snow. At the core of our attack is a differentiable renderer that consistently integrates photorealistic snowflakes with realistic motion into the 3D scene. Through optimization we obtain adversarial snow that significantly impacts the optical flow while being indistinguishable from ordinary snow. Surprisingly, the impact of our novel attack is largest on methods that previously showed a high robustness to small L_p perturbations.



## **24. UKP-SQuARE v2: Explainability and Adversarial Attacks for Trustworthy QA**

cs.CL

Accepted at AACL 2022 as Demo Paper

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2208.09316v3) [paper-pdf](http://arxiv.org/pdf/2208.09316v3)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstract**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.



## **25. Analyzing the Robustness of Decentralized Horizontal and Vertical Federated Learning Architectures in a Non-IID Scenario**

cs.LG

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11061v1) [paper-pdf](http://arxiv.org/pdf/2210.11061v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Enrique Tomás Martínez Beltrán, Daniel Demeter, Gérôme Bovet, Gregorio Martínez Pérez, Burkhard Stiller

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine and deep learning models while protecting data privacy. However, the FL paradigm still presents drawbacks affecting its trustworthiness since malicious participants could launch adversarial attacks against the training process. Related work has studied the robustness of horizontal FL scenarios under different attacks. However, there is a lack of work evaluating the robustness of decentralized vertical FL and comparing it with horizontal FL architectures affected by adversarial attacks. Thus, this work proposes three decentralized FL architectures, one for horizontal and two for vertical scenarios, namely HoriChain, VertiChain, and VertiComb. These architectures present different neural networks and training protocols suitable for horizontal and vertical scenarios. Then, a decentralized, privacy-preserving, and federated use case with non-IID data to classify handwritten digits is deployed to evaluate the performance of the three architectures. Finally, a set of experiments computes and compares the robustness of the proposed architectures when they are affected by different data poisoning based on image watermarks and gradient poisoning adversarial attacks. The experiments show that even though particular configurations of both attacks can destroy the classification performance of the architectures, HoriChain is the most robust one.



## **26. Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame**

cs.CV

Accepted at IEEE Transactions on Image Processing (TIP), 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2204.13004v2) [paper-pdf](http://arxiv.org/pdf/2204.13004v2)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstract**: Person detection has attracted great attention in the computer vision area and is an imperative element in human-centric computer vision. Although the predictive performances of person detection networks have been improved dramatically, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the person detection network in safety-critical applications such as autonomous driving and security systems. Despite the necessity of countering adversarial patch attacks, very few efforts have been dedicated to defending person detection against adversarial patch attack. In this paper, we propose a novel defense strategy that defends against an adversarial patch attack by optimizing a defensive frame for person detection. The defensive frame alleviates the effect of the adversarial patch while maintaining person detection performance with clean person. The proposed defensive frame in the person detection is generated with a competitive learning algorithm which makes an iterative competition between detection threatening module and detection shielding module in person detection. Comprehensive experimental results demonstrate that the proposed method effectively defends person detection against adversarial patch attacks.



## **27. Chaos Theory and Adversarial Robustness**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.13235v1) [paper-pdf](http://arxiv.org/pdf/2210.13235v1)

**Authors**: Jonathan S. Kent

**Abstract**: Neural Networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which Neural Networks are susceptible to or robust against adversarial attacks. Our results show that susceptibility to attack grows significantly with the depth of the model, which has significant safety implications for the design of Neural Networks for production environments. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly, as well as show a clear relationship between our new susceptibility metric and post-attack accuracy.



## **28. Towards Adversarial Attack on Vision-Language Pre-training Models**

cs.LG

Accepted by ACM MM2022. Code is available in GitHub

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2206.09391v2) [paper-pdf](http://arxiv.org/pdf/2206.09391v2)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstract**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios. Code is available at https://github.com/adversarial-for-goodness/Co-Attack.



## **29. Rewriting Meaningful Sentences via Conditional BERT Sampling and an application on fooling text classifiers**

cs.CL

Please see an updated version of this paper at arXiv:2104.08453

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2010.11869v2) [paper-pdf](http://arxiv.org/pdf/2010.11869v2)

**Authors**: Lei Xu, Ivan Ramirez, Kalyan Veeramachaneni

**Abstract**: Most adversarial attack methods that are designed to deceive a text classifier change the text classifier's prediction by modifying a few words or characters. Few try to attack classifiers by rewriting a whole sentence, due to the difficulties inherent in sentence-level rephrasing as well as the problem of setting the criteria for legitimate rewriting.   In this paper, we explore the problem of creating adversarial examples with sentence-level rewriting. We design a new sampling method, named ParaphraseSampler, to efficiently rewrite the original sentence in multiple ways. Then we propose a new criteria for modification, called a sentence-level threaten model. This criteria allows for both word- and sentence-level changes, and can be adjusted independently in two dimensions: semantic similarity and grammatical quality. Experimental results show that many of these rewritten sentences are misclassified by the classifier. On all 6 datasets, our ParaphraseSampler achieves a better attack success rate than our baseline.



## **30. R&R: Metric-guided Adversarial Sentence Generation**

cs.CL

Accepted to Finding of AACL-IJCNLP2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2104.08453v3) [paper-pdf](http://arxiv.org/pdf/2104.08453v3)

**Authors**: Lei Xu, Alfredo Cuesta-Infante, Laure Berti-Equille, Kalyan Veeramachaneni

**Abstract**: Adversarial examples are helpful for analyzing and improving the robustness of text classifiers. Generating high-quality adversarial examples is a challenging task as it requires generating fluent adversarial sentences that are semantically similar to the original sentences and preserve the original labels, while causing the classifier to misclassify them. Existing methods prioritize misclassification by maximizing each perturbation's effectiveness at misleading a text classifier; thus, the generated adversarial examples fall short in terms of fluency and similarity. In this paper, we propose a rewrite and rollback (R&R) framework for adversarial attack. It improves the quality of adversarial examples by optimizing a critique score which combines the fluency, similarity, and misclassification metrics. R&R generates high-quality adversarial examples by allowing exploration of perturbations that do not have immediate impact on the misclassification metric but can improve fluency and similarity metrics. We evaluate our method on 5 representative datasets and 3 classifier architectures. Our method outperforms current state-of-the-art in attack success rate by +16.2%, +12.8%, and +14.0% on the classifiers respectively. Code is available at https://github.com/DAI-Lab/fibber



## **31. Backdoor Attack and Defense in Federated Generative Adversarial Network-based Medical Image Synthesis**

cs.CV

25 pages, 7 figures. arXiv admin note: text overlap with  arXiv:2207.00762

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10886v1) [paper-pdf](http://arxiv.org/pdf/2210.10886v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstract**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research and augment medical datasets. Training generative adversarial neural networks (GANs) usually require large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data while keeping raw data locally. However, given that the FL server cannot access the raw data, it is vulnerable to backdoor attacks, an adversarial by poisoning training data. Most backdoor attack strategies focus on classification models and centralized domains. It is still an open question if the existing backdoor attacks can affect GAN training and, if so, how to defend against the attack in the FL setting. In this work, we investigate the overlooked issue of backdoor attacks in federated GANs (FedGANs). The success of this attack is subsequently determined to be the result of some local discriminators overfitting the poisoned data and corrupting the local GAN equilibrium, which then further contaminates other clients when averaging the generator's parameters and yields high generator loss. Therefore, we proposed FedDetect, an efficient and effective way of defending against the backdoor attack in the FL setting, which allows the server to detect the client's adversarial behavior based on their losses and block the malicious clients. Our extensive experiments on two medical datasets with different modalities demonstrate the backdoor attack on FedGANs can result in synthetic images with low fidelity. After detecting and suppressing the detected malicious clients using the proposed defense strategy, we show that FedGANs can synthesize high-quality medical datasets (with labels) for data augmentation to improve classification models' performance.



## **32. On the Perils of Cascading Robust Classifiers**

cs.LG

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2206.00278v2) [paper-pdf](http://arxiv.org/pdf/2206.00278v2)

**Authors**: Ravi Mangal, Zifan Wang, Chi Zhang, Klas Leino, Corina Pasareanu, Matt Fredrikson

**Abstract**: Ensembling certifiably robust neural networks is a promising approach for improving the \emph{certified robust accuracy} of neural models. Black-box ensembles that assume only query-access to the constituent models (and their robustness certifiers) during prediction are particularly attractive due to their modular structure. Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. However, we show that the robustness certifier used by a cascading ensemble is unsound. That is, when a cascading ensemble is certified as locally robust at an input $x$ (with respect to $\epsilon$), there can be inputs $x'$ in the $\epsilon$-ball centered at $x$, such that the cascade's prediction at $x'$ is different from $x$ and thus the ensemble is not locally robust. Our theoretical findings are accompanied by empirical results that further demonstrate this unsoundness. We present \emph{cascade attack} (CasA), an adversarial attack against cascading ensembles, and show that: (1) there exists an adversarial input for up to 88\% of the samples where the ensemble claims to be certifiably robust and accurate; and (2) the accuracy of a cascading ensemble under our attack is as low as 11\% when it claims to be certifiably robust and accurate on 97\% of the test set. Our work reveals a critical pitfall of cascading certifiably robust models by showing that the seemingly beneficial strategy of cascading can actually hurt the robustness of the resulting ensemble. Our code is available at \url{https://github.com/TristaChi/ensembleKW}.



## **33. Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP**

cs.CL

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10683v1) [paper-pdf](http://arxiv.org/pdf/2210.10683v1)

**Authors**: Yangyi Chen, Hongcheng Gao, Ganqu Cui, Fanchao Qi, Longtao Huang, Zhiyuan Liu, Maosong Sun

**Abstract**: Textual adversarial samples play important roles in multiple subfields of NLP research, including security, evaluation, explainability, and data augmentation. However, most work mixes all these roles, obscuring the problem definitions and research goals of the security role that aims to reveal the practical concerns of NLP models. In this paper, we rethink the research paradigm of textual adversarial samples in security scenarios. We discuss the deficiencies in previous work and propose our suggestions that the research on the Security-oriented adversarial NLP (SoadNLP) should: (1) evaluate their methods on security tasks to demonstrate the real-world concerns; (2) consider real-world attackers' goals, instead of developing impractical methods. To this end, we first collect, process, and release a security datasets collection Advbench. Then, we reformalize the task and adjust the emphasis on different goals in SoadNLP. Next, we propose a simple method based on heuristic rules that can easily fulfill the actual adversarial goals to simulate real-world attack methods. We conduct experiments on both the attack and the defense sides on Advbench. Experimental results show that our method has higher practical value, indicating that the research paradigm in SoadNLP may start from our new benchmark. All the code and data of Advbench can be obtained at \url{https://github.com/thunlp/Advbench}.



## **34. Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks**

cs.CR

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2110.08247v2) [paper-pdf](http://arxiv.org/pdf/2110.08247v2)

**Authors**: Yangyi Chen, Fanchao Qi, Hongcheng Gao, Zhiyuan Liu, Maosong Sun

**Abstract**: Backdoor attacks are a kind of emergent security threat in deep learning. After being injected with a backdoor, a deep neural model will behave normally on standard inputs but give adversary-specified predictions once the input contains specific backdoor triggers. In this paper, we find two simple tricks that can make existing textual backdoor attacks much more harmful. The first trick is to add an extra training task to distinguish poisoned and clean data during the training of the victim model, and the second one is to use all the clean training data rather than remove the original clean data corresponding to the poisoned data. These two tricks are universally applicable to different attack models. We conduct experiments in three tough situations including clean data fine-tuning, low-poisoning-rate, and label-consistent attacks. Experimental results show that the two tricks can significantly improve attack performance. This paper exhibits the great potential harmfulness of backdoor attacks. All the code and data can be obtained at \url{https://github.com/thunlp/StyleAttack}.



## **35. Few-shot Transferable Robust Representation Learning via Bilevel Attacks**

cs.LG

*Equal contribution. Author ordering determined by coin flip

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10485v1) [paper-pdf](http://arxiv.org/pdf/2210.10485v1)

**Authors**: Minseon Kim, Hyeonjeong Ha, Sung Ju Hwang

**Abstract**: Existing adversarial learning methods for enhancing the robustness of deep neural networks assume the availability of a large amount of data from which we can generate adversarial examples. However, in an adversarial meta-learning setting, the model needs to train with only a few adversarial examples to learn a robust model for unseen tasks, which is a very difficult goal to achieve. Further, learning transferable robust representations for unseen domains is a difficult problem even with a large amount of data. To tackle such a challenge, we propose a novel adversarial self-supervised meta-learning framework with bilevel attacks which aims to learn robust representations that can generalize across tasks and domains. Specifically, in the inner loop, we update the parameters of the given encoder by taking inner gradient steps using two different sets of augmented samples, and generate adversarial examples for each view by maximizing the instance classification loss. Then, in the outer loop, we meta-learn the encoder parameter to maximize the agreement between the two adversarial examples, which enables it to learn robust representations. We experimentally validate the effectiveness of our approach on unseen domain adaptation tasks, on which it achieves impressive performance. Specifically, our method significantly outperforms the state-of-the-art meta-adversarial learning methods on few-shot learning tasks, as well as self-supervised learning baselines in standard learning settings with large-scale datasets.



## **36. Emerging Threats in Deep Learning-Based Autonomous Driving: A Comprehensive Survey**

cs.CR

28 pages,10 figures

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.11237v1) [paper-pdf](http://arxiv.org/pdf/2210.11237v1)

**Authors**: Hui Cao, Wenlong Zou, Yinkun Wang, Ting Song, Mengjun Liu

**Abstract**: Since the 2004 DARPA Grand Challenge, the autonomous driving technology has witnessed nearly two decades of rapid development. Particularly, in recent years, with the application of new sensors and deep learning technologies extending to the autonomous field, the development of autonomous driving technology has continued to make breakthroughs. Thus, many carmakers and high-tech giants dedicated to research and system development of autonomous driving. However, as the foundation of autonomous driving, the deep learning technology faces many new security risks. The academic community has proposed deep learning countermeasures against the adversarial examples and AI backdoor, and has introduced them into the autonomous driving field for verification. Deep learning security matters to autonomous driving system security, and then matters to personal safety, which is an issue that deserves attention and research.This paper provides an summary of the concepts, developments and recent research in deep learning security technologies in autonomous driving. Firstly, we briefly introduce the deep learning framework and pipeline in the autonomous driving system, which mainly include the deep learning technologies and algorithms commonly used in this field. Moreover, we focus on the potential security threats of the deep learning based autonomous driving system in each functional layer in turn. We reviews the development of deep learning attack technologies to autonomous driving, investigates the State-of-the-Art algorithms, and reveals the potential risks. At last, we provides an outlook on deep learning security in the autonomous driving field and proposes recommendations for building a safe and trustworthy autonomous driving system.



## **37. On the Adversarial Robustness of Mixture of Experts**

cs.LG

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10253v1) [paper-pdf](http://arxiv.org/pdf/2210.10253v1)

**Authors**: Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi, Srinadh Bhojanapalli

**Abstract**: Adversarial robustness is a key desirable property of neural networks. It has been empirically shown to be affected by their sizes, with larger networks being typically more robust. Recently, Bubeck and Sellke proved a lower bound on the Lipschitz constant of functions that fit the training data in terms of their number of parameters. This raises an interesting open question, do -- and can -- functions with more parameters, but not necessarily more computational cost, have better robustness? We study this question for sparse Mixture of Expert models (MoEs), that make it possible to scale up the model size for a roughly constant computational cost. We theoretically show that under certain conditions on the routing and the structure of the data, MoEs can have significantly smaller Lipschitz constants than their dense counterparts. The robustness of MoEs can suffer when the highest weighted experts for an input implement sufficiently different functions. We next empirically evaluate the robustness of MoEs on ImageNet using adversarial attacks and show they are indeed more robust than dense models with the same computational cost. We make key observations showing the robustness of MoEs to the choice of experts, highlighting the redundancy of experts in models trained in practice.



## **38. Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization**

cs.LG

Accepted and to appear at ICML 2019

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/1905.06635v2) [paper-pdf](http://arxiv.org/pdf/1905.06635v2)

**Authors**: Seungyong Moon, Gaon An, Hyun Oh Song

**Abstract**: Solving for adversarial examples with projected gradient descent has been demonstrated to be highly effective in fooling the neural network based classifiers. However, in the black-box setting, the attacker is limited only to the query access to the network and solving for a successful adversarial example becomes much more difficult. To this end, recent methods aim at estimating the true gradient signal based on the input queries but at the cost of excessive queries. We propose an efficient discrete surrogate to the optimization problem which does not require estimating the gradient and consequently becomes free of the first order update hyperparameters to tune. Our experiments on Cifar-10 and ImageNet show the state of the art black-box attack performance with significant reduction in the required queries compared to a number of recently proposed methods. The source code is available at https://github.com/snu-mllab/parsimonious-blackbox-attack.



## **39. Scaling Adversarial Training to Large Perturbation Bounds**

cs.LG

ECCV 2022

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09852v1) [paper-pdf](http://arxiv.org/pdf/2210.09852v1)

**Authors**: Sravanti Addepalli, Samyak Jain, Gaurang Sriramanan, R. Venkatesh Babu

**Abstract**: The vulnerability of Deep Neural Networks to Adversarial Attacks has fuelled research towards building robust models. While most Adversarial Training algorithms aim at defending attacks constrained within low magnitude Lp norm bounds, real-world adversaries are not limited by such constraints. In this work, we aim to achieve adversarial robustness within larger bounds, against perturbations that may be perceptible, but do not change human (or Oracle) prediction. The presence of images that flip Oracle predictions and those that do not makes this a challenging setting for adversarial robustness. We discuss the ideal goals of an adversarial defense algorithm beyond perceptual limits, and further highlight the shortcomings of naively extending existing training algorithms to higher perturbation bounds. In order to overcome these shortcomings, we propose a novel defense, Oracle-Aligned Adversarial Training (OA-AT), to align the predictions of the network with that of an Oracle during adversarial training. The proposed approach achieves state-of-the-art performance at large epsilon bounds (such as an L-inf bound of 16/255 on CIFAR-10) while outperforming existing defenses (AWP, TRADES, PGD-AT) at standard bounds (8/255) as well.



## **40. Provably Robust Detection of Out-of-distribution Data (almost) for free**

cs.LG

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2106.04260v2) [paper-pdf](http://arxiv.org/pdf/2106.04260v2)

**Authors**: Alexander Meinke, Julian Bitterwolf, Matthias Hein

**Abstract**: The application of machine learning in safety-critical systems requires a reliable assessment of uncertainty. However, deep neural networks are known to produce highly overconfident predictions on out-of-distribution (OOD) data. Even if trained to be non-confident on OOD data, one can still adversarially manipulate OOD data so that the classifier again assigns high confidence to the manipulated samples. We show that two previously published defenses can be broken by better adapted attacks, highlighting the importance of robustness guarantees around OOD data. Since the existing method for this task is hard to train and significantly limits accuracy, we construct a classifier that can simultaneously achieve provably adversarially robust OOD detection and high clean accuracy. Moreover, by slightly modifying the classifier's architecture our method provably avoids the asymptotic overconfidence problem of standard neural networks. We provide code for all our experiments.



## **41. ROSE: Robust Selective Fine-tuning for Pre-trained Language Models**

cs.CL

Accepted to EMNLP 2022. Code is available at  https://github.com/jiangllan/ROSE

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09658v1) [paper-pdf](http://arxiv.org/pdf/2210.09658v1)

**Authors**: Lan Jiang, Hao Zhou, Yankai Lin, Peng Li, Jie Zhou, Rui Jiang

**Abstract**: Even though the large-scale language models have achieved excellent performances, they suffer from various adversarial attacks. A large body of defense methods has been proposed. However, they are still limited due to redundant attack search spaces and the inability to defend against various types of attacks. In this work, we present a novel fine-tuning approach called \textbf{RO}bust \textbf{SE}letive fine-tuning (\textbf{ROSE}) to address this issue. ROSE conducts selective updates when adapting pre-trained models to downstream tasks, filtering out invaluable and unrobust updates of parameters. Specifically, we propose two strategies: the first-order and second-order ROSE for selecting target robust parameters. The experimental results show that ROSE achieves significant improvements in adversarial robustness on various downstream NLP tasks, and the ensemble method even surpasses both variants above. Furthermore, ROSE can be easily incorporated into existing fine-tuning methods to improve their adversarial robustness further. The empirical analysis confirms that ROSE eliminates unrobust spurious updates during fine-tuning, leading to solutions corresponding to flatter and wider optima than the conventional method. Code is available at \url{https://github.com/jiangllan/ROSE}.



## **42. Analysis of Master Vein Attacks on Finger Vein Recognition Systems**

cs.CV

Accepted to be Published in Proceedings of the IEEE/CVF Winter  Conference on Applications of Computer Vision (WACV) 2023

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.10667v1) [paper-pdf](http://arxiv.org/pdf/2210.10667v1)

**Authors**: Huy H. Nguyen, Trung-Nghia Le, Junichi Yamagishi, Isao Echizen

**Abstract**: Finger vein recognition (FVR) systems have been commercially used, especially in ATMs, for customer verification. Thus, it is essential to measure their robustness against various attack methods, especially when a hand-crafted FVR system is used without any countermeasure methods. In this paper, we are the first in the literature to introduce master vein attacks in which we craft a vein-looking image so that it can falsely match with as many identities as possible by the FVR systems. We present two methods for generating master veins for use in attacking these systems. The first uses an adaptation of the latent variable evolution algorithm with a proposed generative model (a multi-stage combination of beta-VAE and WGAN-GP models). The second uses an adversarial machine learning attack method to attack a strong surrogate CNN-based recognition system. The two methods can be easily combined to boost their attack ability. Experimental results demonstrated that the proposed methods alone and together achieved false acceptance rates up to 73.29% and 88.79%, respectively, against Miura's hand-crafted FVR system. We also point out that Miura's system is easily compromised by non-vein-looking samples generated by a WGAN-GP model with false acceptance rates up to 94.21%. The results raise the alarm about the robustness of such systems and suggest that master vein attacks should be considered an important security measure.



## **43. Making Split Learning Resilient to Label Leakage by Potential Energy Loss**

cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09617v1) [paper-pdf](http://arxiv.org/pdf/2210.09617v1)

**Authors**: Fei Zheng, Chaochao Chen, Binhui Yao, Xiaolin Zheng

**Abstract**: As a practical privacy-preserving learning method, split learning has drawn much attention in academia and industry. However, its security is constantly being questioned since the intermediate results are shared during training and inference. In this paper, we focus on the privacy leakage problem caused by the trained split model, i.e., the attacker can use a few labeled samples to fine-tune the bottom model, and gets quite good performance. To prevent such kind of privacy leakage, we propose the potential energy loss to make the output of the bottom model become a more `complicated' distribution, by pushing outputs of the same class towards the decision boundary. Therefore, the adversary suffers a large generalization error when fine-tuning the bottom model with only a few leaked labeled samples. Experiment results show that our method significantly lowers the attacker's fine-tuning accuracy, making the split model more resilient to label leakage.



## **44. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2208.10224v3) [paper-pdf](http://arxiv.org/pdf/2208.10224v3)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.



## **45. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

cs.LG

Published in NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2202.01181v3) [paper-pdf](http://arxiv.org/pdf/2202.01181v3)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstract**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named Catastrophic Overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. Experimentally they showed that simply adding a random perturbation prior to FGSM (RS-FGSM) could prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed a computationally expensive regularizer (GradAlign) to avoid it. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with \textit{not clipping} is highly effective in avoiding CO for large perturbation radii. We then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous state-of-the-art GradAlign, while achieving 3x speed-up. Code can be found in https://github.com/pdejorge/N-FGSM



## **46. Deepfake Text Detection: Limitations and Opportunities**

cs.CR

Accepted to IEEE S&P 2023; First two authors contributed equally to  this work; 18 pages, 7 figures

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09421v1) [paper-pdf](http://arxiv.org/pdf/2210.09421v1)

**Authors**: Jiameng Pu, Zain Sarwar, Sifat Muhammad Abdullah, Abdullah Rehman, Yoonjin Kim, Parantapa Bhattacharya, Mobin Javed, Bimal Viswanath

**Abstract**: Recent advances in generative models for language have enabled the creation of convincing synthetic text or deepfake text. Prior work has demonstrated the potential for misuse of deepfake text to mislead content consumers. Therefore, deepfake text detection, the task of discriminating between human and machine-generated text, is becoming increasingly critical. Several defenses have been proposed for deepfake text detection. However, we lack a thorough understanding of their real-world applicability. In this paper, we collect deepfake text from 4 online services powered by Transformer-based tools to evaluate the generalization ability of the defenses on content in the wild. We develop several low-cost adversarial attacks, and investigate the robustness of existing defenses against an adaptive attacker. We find that many defenses show significant degradation in performance under our evaluation scenarios compared to their original claimed performance. Our evaluation shows that tapping into the semantic information in the text content is a promising approach for improving the robustness and generalization performance of deepfake text detection schemes.



## **47. Towards Generating Adversarial Examples on Mixed-type Data**

cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09405v1) [paper-pdf](http://arxiv.org/pdf/2210.09405v1)

**Authors**: Han Xu, Menghai Pan, Zhimeng Jiang, Huiyuan Chen, Xiaoting Li, Mahashweta Das, Hao Yang

**Abstract**: The existence of adversarial attacks (or adversarial examples) brings huge concern about the machine learning (ML) model's safety issues. For many safety-critical ML tasks, such as financial forecasting, fraudulent detection, and anomaly detection, the data samples are usually mixed-type, which contain plenty of numerical and categorical features at the same time. However, how to generate adversarial examples with mixed-type data is still seldom studied. In this paper, we propose a novel attack algorithm M-Attack, which can effectively generate adversarial examples in mixed-type data. Based on M-Attack, attackers can attempt to mislead the targeted classification model's prediction, by only slightly perturbing both the numerical and categorical features in the given data samples. More importantly, by adding designed regularizations, our generated adversarial examples can evade potential detection models, which makes the attack indeed insidious. Through extensive empirical studies, we validate the effectiveness and efficiency of our attack method and evaluate the robustness of existing classification models against our proposed attack. The experimental results highlight the feasibility of generating adversarial examples toward machine learning models in real-world applications.



## **48. Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class**

cs.CR

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09194v1) [paper-pdf](http://arxiv.org/pdf/2210.09194v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Ping Li

**Abstract**: In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.



## **49. Adversarial Robustness is at Odds with Lazy Training**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2207.00411v2) [paper-pdf](http://arxiv.org/pdf/2207.00411v2)

**Authors**: Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora

**Abstract**: Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to "lazy training" of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.



## **50. DE-CROP: Data-efficient Certified Robustness for Pretrained Classifiers**

cs.LG

WACV 2023. Project page: https://sites.google.com/view/decrop

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08929v1) [paper-pdf](http://arxiv.org/pdf/2210.08929v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Certified defense using randomized smoothing is a popular technique to provide robustness guarantees for deep neural networks against l2 adversarial attacks. Existing works use this technique to provably secure a pretrained non-robust model by training a custom denoiser network on entire training data. However, access to the training set may be restricted to a handful of data samples due to constraints such as high transmission cost and the proprietary nature of the data. Thus, we formulate a novel problem of "how to certify the robustness of pretrained models using only a few training samples". We observe that training the custom denoiser directly using the existing techniques on limited samples yields poor certification. To overcome this, our proposed approach (DE-CROP) generates class-boundary and interpolated samples corresponding to each training sample, ensuring high diversity in the feature space of the pretrained classifier. We train the denoiser by maximizing the similarity between the denoised output of the generated sample and the original training sample in the classifier's logit space. We also perform distribution level matching using domain discriminator and maximum mean discrepancy that yields further benefit. In white box setup, we obtain significant improvements over the baseline on multiple benchmark datasets and also report similar performance under the challenging black box setup.



