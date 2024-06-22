# Latest Adversarial Attack Papers
**update at 2024-06-22 15:59:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Jailbreaking as a Reward Misspecification Problem**

cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14393v1) [paper-pdf](http://arxiv.org/pdf/2406.14393v1)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.



## **2. On countering adversarial perturbations in graphs using error correcting codes**

cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14245v1) [paper-pdf](http://arxiv.org/pdf/2406.14245v1)

**Authors**: Saif Eddin Jabari

**Abstract**: We consider the problem of a graph subjected to adversarial perturbations, such as those arising from cyber-attacks, where edges are covertly added or removed. The adversarial perturbations occur during the transmission of the graph between a sender and a receiver. To counteract potential perturbations, we explore a repetition coding scheme with sender-assigned binary noise and majority voting on the receiver's end to rectify the graph's structure. Our approach operates without prior knowledge of the attack's characteristics. We provide an analytical derivation of a bound on the number of repetitions needed to satisfy probabilistic constraints on the quality of the reconstructed graph. We show that the method can accurately decode graphs that were subjected to non-random edge removal, namely, those connected to vertices with the highest eigenvector centrality, in addition to random addition and removal of edges by the attacker.



## **3. Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2311.06942v2) [paper-pdf](http://arxiv.org/pdf/2311.06942v2)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of contractive dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.



## **4. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

cs.CR

(Last update) Stochastic investment models and a Bayesian approach to  better modeling of uncertainty : adversarial machine learning or Stochastic  market. arXiv admin note: substantial text overlap with arXiv:2402.05967

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.10719v2) [paper-pdf](http://arxiv.org/pdf/2406.10719v2)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.



## **5. Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders**

cs.IR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2305.04694v2) [paper-pdf](http://arxiv.org/pdf/2305.04694v2)

**Authors**: Sulthana Shams, Douglas Leith

**Abstract**: In practice, users of a Recommender System (RS) fall into a few clusters based on their preferences. In this work, we conduct a systematic study on user-cluster targeted data poisoning attacks on Matrix Factorisation (MF) based RS, where an adversary injects fake users with falsely crafted user-item feedback to promote an item to a specific user cluster. We analyse how user and item feature matrices change after data poisoning attacks and identify the factors that influence the effectiveness of the attack on these feature matrices. We demonstrate that the adversary can easily target specific user clusters with minimal effort and that some items are more susceptible to attacks than others. Our theoretical analysis has been validated by the experimental results obtained from two real-world datasets. Our observations from the study could serve as a motivating point to design a more robust RS.



## **6. A Survey of Fragile Model Watermarking**

cs.CR

Submitted Expert Systems with Applications

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.04809v3) [paper-pdf](http://arxiv.org/pdf/2406.04809v3)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.



## **7. Explainable AI Security: Exploring Robustness of Graph Neural Networks to Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.13920v1) [paper-pdf](http://arxiv.org/pdf/2406.13920v1)

**Authors**: Tao Wu, Canyixing Cui, Xingping Xian, Shaojie Qiao, Chao Wang, Lin Yuan, Shui Yu

**Abstract**: Graph neural networks (GNNs) have achieved tremendous success, but recent studies have shown that GNNs are vulnerable to adversarial attacks, which significantly hinders their use in safety-critical scenarios. Therefore, the design of robust GNNs has attracted increasing attention. However, existing research has mainly been conducted via experimental trial and error, and thus far, there remains a lack of a comprehensive understanding of the vulnerability of GNNs. To address this limitation, we systematically investigate the adversarial robustness of GNNs by considering graph data patterns, model-specific factors, and the transferability of adversarial examples. Through extensive experiments, a set of principled guidelines is obtained for improving the adversarial robustness of GNNs, for example: (i) rather than highly regular graphs, the training graph data with diverse structural patterns is crucial for model robustness, which is consistent with the concept of adversarial training; (ii) the large model capacity of GNNs with sufficient training data has a positive effect on model robustness, and only a small percentage of neurons in GNNs are affected by adversarial attacks; (iii) adversarial transfer is not symmetric and the adversarial examples produced by the small-capacity model have stronger adversarial transferability. This work illuminates the vulnerabilities of GNNs and opens many promising avenues for designing robust GNNs.



## **8. RLHFPoison: Reward Poisoning Attack for Reinforcement Learning with Human Feedback in Large Language Models**

cs.AI

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.09641v2) [paper-pdf](http://arxiv.org/pdf/2311.09641v2)

**Authors**: Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, Chaowei Xiao

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) is a methodology designed to align Large Language Models (LLMs) with human preferences, playing an important role in LLMs alignment. Despite its advantages, RLHF relies on human annotators to rank the text, which can introduce potential security vulnerabilities if any adversarial annotator (i.e., attackers) manipulates the ranking score by up-ranking any malicious text to steer the LLM adversarially. To assess the red-teaming of RLHF against human preference data poisoning, we propose RankPoison, a poisoning attack method on candidates' selection of preference rank flipping to reach certain malicious behaviors (e.g., generating longer sequences, which can increase the computational cost). With poisoned dataset generated by RankPoison, we can perform poisoning attacks on LLMs to generate longer tokens without hurting the original safety alignment performance. Moreover, applying RankPoison, we also successfully implement a backdoor attack where LLMs can generate longer answers under questions with the trigger word. Our findings highlight critical security challenges in RLHF, underscoring the necessity for more robust alignment methods for LLMs.



## **9. Benchmarking Unsupervised Online IDS for Masquerade Attacks in CAN**

cs.CR

15 pages, 9 figures, 3 tables

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13778v1) [paper-pdf](http://arxiv.org/pdf/2406.13778v1)

**Authors**: Pablo Moriano, Steven C. Hespeler, Mingyan Li, Robert A. Bridges

**Abstract**: Vehicular controller area networks (CANs) are susceptible to masquerade attacks by malicious adversaries. In masquerade attacks, adversaries silence a targeted ID and then send malicious frames with forged content at the expected timing of benign frames. As masquerade attacks could seriously harm vehicle functionality and are the stealthiest attacks to detect in CAN, recent work has devoted attention to compare frameworks for detecting masquerade attacks in CAN. However, most existing works report offline evaluations using CAN logs already collected using simulations that do not comply with domain's real-time constraints. Here we contribute to advance the state of the art by introducing a benchmark study of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing benchmarks in that we analyze the effect of controlling streaming data conditions in a sliding window setting. In doing so, we use realistic masquerade attacks being replayed from the ROAD dataset. We show that although benchmarked IDS are not effective at detecting every attack type, the method that relies on detecting changes at the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead. We discuss limitations, open challenges, and how the benchmarked methods can be used for practical unsupervised online CAN IDS for masquerade attacks.



## **10. Confidence Is All You Need for MI Attacks**

cs.LG

2 pages, 1 figure

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2311.15373v2) [paper-pdf](http://arxiv.org/pdf/2311.15373v2)

**Authors**: Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

**Abstract**: In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point's membership in a model's training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being 'fit' to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine learning model. These confidence values provide a probabilistic measure of the model's certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.



## **11. Fooling Polarization-based Vision using Locally Controllable Polarizing Projection**

cs.CV

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2303.17890v2) [paper-pdf](http://arxiv.org/pdf/2303.17890v2)

**Authors**: Zhuoxiao Li, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng

**Abstract**: Polarization is a fundamental property of light that encodes abundant information regarding surface shape, material, illumination and viewing geometry. The computer vision community has witnessed a blossom of polarization-based vision applications, such as reflection removal, shape-from-polarization, transparent object segmentation and color constancy, partially due to the emergence of single-chip mono/color polarization sensors that make polarization data acquisition easier than ever. However, is polarization-based vision vulnerable to adversarial attacks? If so, is that possible to realize these adversarial attacks in the physical world, without being perceived by human eyes? In this paper, we warn the community of the vulnerability of polarization-based vision, which can be more serious than RGB-based vision. By adapting a commercial LCD projector, we achieve locally controllable polarizing projection, which is successfully utilized to fool state-of-the-art polarization-based vision algorithms for glass segmentation and color constancy. Compared with existing physical attacks on RGB-based vision, which always suffer from the trade-off between attack efficacy and eye conceivability, the adversarial attackers based on polarizing projection are contact-free and visually imperceptible, since naked human eyes can rarely perceive the difference of viciously manipulated polarizing light and ordinary illumination. This poses unprecedented risks on polarization-based vision, both in the monochromatic and trichromatic domain, for which due attentions should be paid and counter measures be considered.



## **12. Bayes' capacity as a measure for reconstruction attacks in federated learning**

cs.LG

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13569v1) [paper-pdf](http://arxiv.org/pdf/2406.13569v1)

**Authors**: Sayan Biswas, Mark Dras, Pedro Faustini, Natasha Fernandes, Annabelle McIver, Catuscia Palamidessi, Parastoo Sadeghi

**Abstract**: Within the machine learning community, reconstruction attacks are a principal attack of concern and have been identified even in federated learning, which was designed with privacy preservation in mind. In federated learning, it has been shown that an adversary with knowledge of the machine learning architecture is able to infer the exact value of a training element given an observation of the weight updates performed during stochastic gradient descent. In response to these threats, the privacy community recommends the use of differential privacy in the stochastic gradient descent algorithm, termed DP-SGD. However, DP has not yet been formally established as an effective countermeasure against reconstruction attacks. In this paper, we formalise the reconstruction threat model using the information-theoretic framework of quantitative information flow. We show that the Bayes' capacity, related to the Sibson mutual information of order infinity, represents a tight upper bound on the leakage of the DP-SGD algorithm to an adversary interested in performing a reconstruction attack. We provide empirical results demonstrating the effectiveness of this measure for comparing mechanisms against reconstruction threats.



## **13. GraphMU: Repairing Robustness of Graph Neural Networks via Machine Unlearning**

cs.SI

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13499v1) [paper-pdf](http://arxiv.org/pdf/2406.13499v1)

**Authors**: Tao Wu, Xinwen Cao, Chao Wang, Shaojie Qiao, Xingping Xian, Lin Yuan, Canyixing Cui, Yanbing Liu

**Abstract**: Graph Neural Networks (GNNs) have demonstrated significant application potential in various fields. However, GNNs are still vulnerable to adversarial attacks. Numerous adversarial defense methods on GNNs are proposed to address the problem of adversarial attacks. However, these methods can only serve as a defense before poisoning, but cannot repair poisoned GNN. Therefore, there is an urgent need for a method to repair poisoned GNN. In this paper, we address this gap by introducing the novel concept of model repair for GNNs. We propose a repair framework, Repairing Robustness of Graph Neural Networks via Machine Unlearning (GraphMU), which aims to fine-tune poisoned GNN to forget adversarial samples without the need for complete retraining. We also introduce a unlearning validation method to ensure that our approach effectively forget specified poisoned data. To evaluate the effectiveness of GraphMU, we explore three fine-tuned subgraph construction scenarios based on the available perturbation information: (i) Known Perturbation Ratios, (ii) Known Complete Knowledge of Perturbations, and (iii) Unknown any Knowledge of Perturbations. Our extensive experiments, conducted across four citation datasets and four adversarial attack scenarios, demonstrate that GraphMU can effectively restore the performance of poisoned GNN.



## **14. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents**

cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13352v1) [paper-pdf](http://arxiv.org/pdf/2406.13352v1)

**Authors**: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr

**Abstract**: AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.



## **15. Textual Unlearning Gives a False Sense of Unlearning**

cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13348v1) [paper-pdf](http://arxiv.org/pdf/2406.13348v1)

**Authors**: Jiacheng Du, Zhibo Wang, Kui Ren

**Abstract**: Language models (LMs) are susceptible to "memorizing" training data, including a large amount of private or copyright-protected content. To safeguard the right to be forgotten (RTBF), machine unlearning has emerged as a promising method for LMs to efficiently "forget" sensitive training content and mitigate knowledge leakage risks. However, despite its good intentions, could the unlearning mechanism be counterproductive? In this paper, we propose the Textual Unlearning Leakage Attack (TULA), where an adversary can infer information about the unlearned data only by accessing the models before and after unlearning. Furthermore, we present variants of TULA in both black-box and white-box scenarios. Through various experimental results, we critically demonstrate that machine unlearning amplifies the risk of knowledge leakage from LMs. Specifically, TULA can increase an adversary's ability to infer membership information about the unlearned data by more than 20% in black-box scenario. Moreover, TULA can even reconstruct the unlearned data directly with more than 60% accuracy with white-box access. Our work is the first to reveal that machine unlearning in LMs can inversely create greater knowledge risks and inspire the development of more secure unlearning mechanisms.



## **16. Blockchain Bribing Attacks and the Efficacy of Counterincentives**

cs.GT

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2402.06352v2) [paper-pdf](http://arxiv.org/pdf/2402.06352v2)

**Authors**: Dimitris Karakostas, Aggelos Kiayias, Thomas Zacharias

**Abstract**: We analyze bribing attacks in Proof-of-Stake distributed ledgers from a game theoretic perspective. In bribing attacks, an adversary offers participants a reward in exchange for instructing them how to behave, with the goal of attacking the protocol's properties. Specifically, our work focuses on adversaries that target blockchain safety. We consider two types of bribing, depending on how the bribes are awarded: i) guided bribing, where the bribe is given as long as the bribed party behaves as instructed; ii) effective bribing, where bribes are conditional on the attack's success, w.r.t. well-defined metrics. We analyze each type of attack in a game theoretic setting and identify relevant equilibria. In guided bribing, we show that the protocol is not an equilibrium and then describe good equilibria, where the attack is unsuccessful, and a negative one, where all parties are bribed such that the attack succeeds. In effective bribing, we show that both the protocol and the "all bribed" setting are equilibria. Using the identified equilibria, we then compute bounds on the Prices of Stability and Anarchy. Our results indicate that additional mitigations are needed for guided bribing, so our analysis concludes with incentive-based mitigation techniques, namely slashing and dilution. Here, we present two positive results, that both render the protocol an equilibrium and achieve maximal welfare for all parties, and a negative result, wherein an attack becomes more plausible if it severely affects the ledger's token's market price.



## **17. Enhancing Cross-Prompt Transferability in Vision-Language Models through Contextual Injection of Target Tokens**

cs.MM

13 pages

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13294v1) [paper-pdf](http://arxiv.org/pdf/2406.13294v1)

**Authors**: Xikang Yang, Xuehai Tang, Fuqing Zhu, Jizhong Han, Songlin Hu

**Abstract**: Vision-language models (VLMs) seamlessly integrate visual and textual data to perform tasks such as image classification, caption generation, and visual question answering. However, adversarial images often struggle to deceive all prompts effectively in the context of cross-prompt migration attacks, as the probability distribution of the tokens in these images tends to favor the semantics of the original image rather than the target tokens. To address this challenge, we propose a Contextual-Injection Attack (CIA) that employs gradient-based perturbation to inject target tokens into both visual and textual contexts, thereby improving the probability distribution of the target tokens. By shifting the contextual semantics towards the target tokens instead of the original image semantics, CIA enhances the cross-prompt transferability of adversarial images.Extensive experiments on the BLIP2, InstructBLIP, and LLaVA models show that CIA outperforms existing methods in cross-prompt transferability, demonstrating its potential for more effective adversarial strategies in VLMs.



## **18. Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation**

cs.LG

8 pages, 5 figures, 3 tables, to be published in ICML: DMLR workshop

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13283v1) [paper-pdf](http://arxiv.org/pdf/2406.13283v1)

**Authors**: Björn Nieth, Thomas Altstidl, Leo Schwinn, Björn Eskofier

**Abstract**: Their vulnerability to small, imperceptible attacks limits the adoption of deep learning models to real-world systems. Adversarial training has proven to be one of the most promising strategies against these attacks, at the expense of a substantial increase in training time. With the ongoing trend of integrating large-scale synthetic data this is only expected to increase even further. Thus, the need for data-centric approaches that reduce the number of training samples while maintaining accuracy and robustness arises. While data pruning and active learning are prominent research topics in deep learning, they are as of now largely unexplored in the adversarial training literature. We address this gap and propose a new data pruning strategy based on extrapolating data importance scores from a small set of data to a larger set. In an empirical evaluation, we demonstrate that extrapolation-based pruning can efficiently reduce dataset size while maintaining robustness.



## **19. AGSOA:Graph Neural Network Targeted Attack Based on Average Gradient and Structure Optimization**

cs.LG

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2406.13228v1) [paper-pdf](http://arxiv.org/pdf/2406.13228v1)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks(GNNs) are vulnerable to adversarial attack that cause performance degradation by adding small perturbations to the graph. Gradient-based attacks are one of the most commonly used methods and have achieved good performance in many attack scenarios. However, current gradient attacks face the problems of easy to fall into local optima and poor attack invisibility. Specifically, most gradient attacks use greedy strategies to generate perturbations, which tend to fall into local optima leading to underperformance of the attack. In addition, many attacks only consider the effectiveness of the attack and ignore the invisibility of the attack, making the attacks easily exposed leading to failure. To address the above problems, this paper proposes an attack on GNNs, called AGSOA, which consists of an average gradient calculation and a structre optimization module. In the average gradient calculation module, we compute the average of the gradient information over all moments to guide the attack to generate perturbed edges, which stabilizes the direction of the attack update and gets rid of undesirable local maxima. In the structure optimization module, we calculate the similarity and homogeneity of the target node's with other nodes to adjust the graph structure so as to improve the invisibility and transferability of the attack. Extensive experiments on three commonly used datasets show that AGSOA improves the misclassification rate by 2$\%$-8$\%$ compared to other state-of-the-art models.



## **20. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

cs.CR

**SubmitDate**: 2024-06-19    [abs](http://arxiv.org/abs/2404.06721v3) [paper-pdf](http://arxiv.org/pdf/2404.06721v3)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.



## **21. NoiSec: Harnessing Noise for Security against Adversarial and Backdoor Attacks**

cs.LG

20 pages, 7 figures

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13073v1) [paper-pdf](http://arxiv.org/pdf/2406.13073v1)

**Authors**: Md Hasan Shahriar, Ning Wang, Y. Thomas Hou, Wenjing Lou

**Abstract**: The exponential adoption of machine learning (ML) is propelling the world into a future of intelligent automation and data-driven solutions. However, the proliferation of malicious data manipulation attacks against ML, namely adversarial and backdoor attacks, jeopardizes its reliability in safety-critical applications. The existing detection methods against such attacks are built upon assumptions, limiting them in diverse practical scenarios. Thus, motivated by the need for a more robust and unified defense mechanism, we investigate the shared traits of adversarial and backdoor attacks and propose NoiSec that leverages solely the noise, the foundational root cause of such attacks, to detect any malicious data alterations. NoiSec is a reconstruction-based detector that disentangles the noise from the test input, extracts the underlying features from the noise, and leverages them to recognize systematic malicious manipulation. Experimental evaluations conducted on the CIFAR10 dataset demonstrate the efficacy of NoiSec, achieving AUROC scores exceeding 0.954 and 0.852 under white-box and black-box adversarial attacks, respectively, and 0.992 against backdoor attacks. Notably, NoiSec maintains a high detection performance, keeping the false positive rate within only 1\%. Comparative analyses against MagNet-based baselines reveal NoiSec's superior performance across various attack scenarios.



## **22. MaskPure: Improving Defense Against Text Adversaries with Stochastic Purification**

cs.LG

15 pages, 1 figure, in the proceedings of The 29th International  Conference on Natural Language & Information Systems (NLDB 2024)

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13066v1) [paper-pdf](http://arxiv.org/pdf/2406.13066v1)

**Authors**: Harrison Gietz, Jugal Kalita

**Abstract**: The improvement of language model robustness, including successful defense against adversarial attacks, remains an open problem. In computer vision settings, the stochastic noising and de-noising process provided by diffusion models has proven useful for purifying input images, thus improving model robustness against adversarial attacks. Similarly, some initial work has explored the use of random noising and de-noising to mitigate adversarial attacks in an NLP setting, but improving the quality and efficiency of these methods is necessary for them to remain competitive. We extend upon methods of input text purification that are inspired by diffusion processes, which randomly mask and refill portions of the input text before classification. Our novel method, MaskPure, exceeds or matches robustness compared to other contemporary defenses, while also requiring no adversarial classifier training and without assuming knowledge of the attack type. In addition, we show that MaskPure is provably certifiably robust. To our knowledge, MaskPure is the first stochastic-purification method with demonstrated success against both character-level and word-level attacks, indicating the generalizable and promising nature of stochastic denoising defenses. In summary: the MaskPure algorithm bridges literature on the current strongest certifiable and empirical adversarial defense methods, showing that both theoretical and practical robustness can be obtained together. Code is available on GitHub at https://github.com/hubarruby/MaskPure.



## **23. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

math.OC

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.13023v1) [paper-pdf](http://arxiv.org/pdf/2406.13023v1)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.



## **24. Can Go AIs be adversarially robust?**

cs.LG

67 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12843v1) [paper-pdf](http://arxiv.org/pdf/2406.12843v1)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs like KataGo can be defeated by simple adversarial strategies. In this paper, we study if simple defenses can improve KataGo's worst-case performance. We test three natural defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that some of these defenses are able to protect against previously discovered attacks. Unfortunately, we also find that none of these defenses are able to withstand adaptive attacks. In particular, we are able to train new adversaries that reliably defeat our defended agents by causing them to blunder in ways humans would not. Our results suggest that building robust AI systems is challenging even in narrow domains such as Go. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.



## **25. Adversarial Attacks on Multimodal Agents**

cs.LG

19 pages

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12814v1) [paper-pdf](http://arxiv.org/pdf/2406.12814v1)

**Authors**: Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan

**Abstract**: Vision-enabled language models (VLMs) are now used to build autonomous multimodal agents capable of taking actions in real environments. In this paper, we show that multimodal agents raise new safety risks, even though attacking agents is more challenging than prior attacks due to limited access to and knowledge about the environment. Our attacks use adversarial text strings to guide gradient-based perturbation over one trigger image in the environment: (1) our captioner attack attacks white-box captioners if they are used to process images into captions as additional inputs to the VLM; (2) our CLIP attack attacks a set of CLIP models jointly, which can transfer to proprietary VLMs. To evaluate the attacks, we curated VisualWebArena-Adv, a set of adversarial tasks based on VisualWebArena, an environment for web-based multimodal agent tasks. Within an L-infinity norm of $16/256$ on a single image, the captioner attack can make a captioner-augmented GPT-4V agent execute the adversarial goals with a 75% success rate. When we remove the captioner or use GPT-4V to generate its own captions, the CLIP attack can achieve success rates of 21% and 43%, respectively. Experiments on agents based on other VLMs, such as Gemini-1.5, Claude-3, and GPT-4o, show interesting differences in their robustness. Further analysis reveals several key factors contributing to the attack's success, and we also discuss the implications for defenses as well. Project page: https://chenwu.io/attack-agent Code and data: https://github.com/ChenWu98/agent-attack



## **26. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B),  jailbreak artifacts for all attacks are available, evaluation of  generalization to a different judge (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots over iterations, ablation on the suffix length  for random search), improved exposition of the paper, examples of jailbroken  generation

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.02151v2) [paper-pdf](http://arxiv.org/pdf/2404.02151v2)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token ``Sure''), potentially with multiple restarts. In this way, we achieve nearly 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.



## **27. UIFV: Data Reconstruction Attack in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12588v1) [paper-pdf](http://arxiv.org/pdf/2406.12588v1)

**Authors**: Jirui Yang, Peng Chen, Zhihui Lu, Qiang Duan, Yubing Bao

**Abstract**: Vertical Federated Learning (VFL) facilitates collaborative machine learning without the need for participants to share raw private data. However, recent studies have revealed privacy risks where adversaries might reconstruct sensitive features through data leakage during the learning process. Although data reconstruction methods based on gradient or model information are somewhat effective, they reveal limitations in VFL application scenarios. This is because these traditional methods heavily rely on specific model structures and/or have strict limitations on application scenarios. To address this, our study introduces the Unified InverNet Framework into VFL, which yields a novel and flexible approach (dubbed UIFV) that leverages intermediate feature data to reconstruct original data, instead of relying on gradients or model details. The intermediate feature data is the feature exchanged by different participants during the inference phase of VFL. Experiments on four datasets demonstrate that our methods significantly outperform state-of-the-art techniques in attack precision. Our work exposes severe privacy vulnerabilities within VFL systems that pose real threats to practical VFL applications and thus confirms the necessity of further enhancing privacy protection in the VFL architecture.



## **28. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2401.07867v2) [paper-pdf](http://arxiv.org/pdf/2401.07867v2)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of recent Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause evasion of automated detection in all tested languages, where homoglyph attacks are especially successful. However, some of the AO methods severely damaged the text, making it no longer readable or easily recognizable by humans (e.g., changed language, weird characters).



## **29. Adversarial Attacks on Large Language Models in Medicine**

cs.AI

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12259v1) [paper-pdf](http://arxiv.org/pdf/2406.12259v1)

**Authors**: Yifan Yang, Qiao Jin, Furong Huang, Zhiyong Lu

**Abstract**: The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks in three medical tasks. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are susceptible to manipulation across multiple tasks. This research further reveals that domain-specific tasks demand more adversarial data in model fine-tuning than general domain tasks for effective attack execution, especially for more capable models. We discover that while integrating adversarial data does not markedly degrade overall model performance on medical benchmarks, it does lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings.



## **30. Privacy-Preserved Neural Graph Databases**

cs.DB

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2312.15591v5) [paper-pdf](http://arxiv.org/pdf/2312.15591v5)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Zihao Wang, Yangqiu Song

**Abstract**: In the era of large language models (LLMs), efficient and accurate data retrieval has become increasingly crucial for the use of domain-specific or private data in the retrieval augmented generation (RAG). Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (GDBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data which can be adaptively trained with LLMs. The usage of neural embedding storage and Complex neural logical Query Answering (CQA) provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the domain-specific or private databases. Malicious attackers can infer more sensitive information in the database using well-designed queries such as from the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training stage due to the privacy concerns. In this work, we propose a privacy-preserved neural graph database (P-NGDB) framework to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to enforce the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries.



## **31. Robust Text Classification: Analyzing Prototype-Based Networks**

cs.CL

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2311.06647v2) [paper-pdf](http://arxiv.org/pdf/2311.06647v2)

**Authors**: Zhivar Sourati, Darshan Deshpande, Filip Ilievski, Kiril Gashteovski, Sascha Saralajew

**Abstract**: Downstream applications often require text classification models to be accurate and robust. While the accuracy of the state-of-the-art Language Models (LMs) approximates human performance, they often exhibit a drop in performance on noisy data found in the real world. This lack of robustness can be concerning, as even small perturbations in the text, irrelevant to the target task, can cause classifiers to incorrectly change their predictions. A potential solution can be the family of Prototype-Based Networks (PBNs) that classifies examples based on their similarity to prototypical examples of a class (prototypes) and has been shown to be robust to noise for computer vision tasks. In this paper, we study whether the robustness properties of PBNs transfer to text classification tasks under both targeted and static adversarial attack settings. Our results show that PBNs, as a mere architectural variation of vanilla LMs, offer more robustness compared to vanilla LMs under both targeted and static settings. We showcase how PBNs' interpretability can help us to understand PBNs' robustness properties. Finally, our ablation studies reveal the sensitivity of PBNs' robustness to how strictly clustering is done in the training phase, as tighter clustering results in less robust PBNs.



## **32. BadSampler: Harnessing the Power of Catastrophic Forgetting to Poison Byzantine-robust Federated Learning**

cs.CR

In Proceedings of the 30th ACM SIGKDD Conference on Knowledge  Discovery and Data Mining (KDD' 24), August 25-29, 2024, Barcelona, Spain

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2406.12222v1) [paper-pdf](http://arxiv.org/pdf/2406.12222v1)

**Authors**: Yi Liu, Cong Wang, Xingliang Yuan

**Abstract**: Federated Learning (FL) is susceptible to poisoning attacks, wherein compromised clients manipulate the global model by modifying local datasets or sending manipulated model updates. Experienced defenders can readily detect and mitigate the poisoning effects of malicious behaviors using Byzantine-robust aggregation rules. However, the exploration of poisoning attacks in scenarios where such behaviors are absent remains largely unexplored for Byzantine-robust FL. This paper addresses the challenging problem of poisoning Byzantine-robust FL by introducing catastrophic forgetting. To fill this gap, we first formally define generalization error and establish its connection to catastrophic forgetting, paving the way for the development of a clean-label data poisoning attack named BadSampler. This attack leverages only clean-label data (i.e., without poisoned data) to poison Byzantine-robust FL and requires the adversary to selectively sample training data with high loss to feed model training and maximize the model's generalization error. We formulate the attack as an optimization problem and present two elegant adversarial sampling strategies, Top-$\kappa$ sampling, and meta-sampling, to approximately solve it. Additionally, our formal error upper bound and time complexity analysis demonstrate that our design can preserve attack utility with high efficiency. Extensive evaluations on two real-world datasets illustrate the effectiveness and performance of our proposed attacks.



## **33. Attack on Scene Flow using Point Clouds**

cs.CV

**SubmitDate**: 2024-06-18    [abs](http://arxiv.org/abs/2404.13621v3) [paper-pdf](http://arxiv.org/pdf/2404.13621v3)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks.



## **34. Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**

cs.LG

ICML 2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2402.02207v2) [paper-pdf](http://arxiv.org/pdf/2402.02207v2)

**Authors**: Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales

**Abstract**: Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.



## **35. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.08298v2) [paper-pdf](http://arxiv.org/pdf/2406.08298v2)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.



## **36. Threat analysis and adversarial model for Smart Grids**

cs.CR

Presented at the Workshop on Attackers and Cyber-Crime Operations  (WACCO). More details available at https://wacco-workshop.org

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11716v1) [paper-pdf](http://arxiv.org/pdf/2406.11716v1)

**Authors**: Javier Sande Ríos, Jesús Canal Sánchez, Carmen Manzano Hernandez, Sergio Pastrana

**Abstract**: The power grid is a critical infrastructure that allows for the efficient and robust generation, transmission, delivery and consumption of electricity. In the recent years, the physical components have been equipped with computing and network devices, which optimizes the operation and maintenance of the grid. The cyber domain of this smart power grid opens a new plethora of threats, which adds to classical threats on the physical domain. Accordingly, different stakeholders including regulation bodies, industry and academy, are making increasing efforts to provide security mechanisms to mitigate and reduce cyber-risks. Despite these efforts, there have been various cyberattacks that have affected the smart grid, leading in some cases to catastrophic consequences, showcasing that the industry might not be prepared for attacks from high profile adversaries. At the same time, recent work shows a lack of agreement among grid practitioners and academic experts on the feasibility and consequences of academic-proposed threats. This is in part due to inadequate simulation models which do not evaluate threats based on attackers full capabilities and goals. To address this gap, in this work we first analyze the main attack surfaces of the smart grid, and then conduct a threat analysis from the adversarial model perspective, including different levels of knowledge, goals, motivations and capabilities. To validate the model, we provide real-world examples of the potential capabilities by studying known vulnerabilities in critical components, and then analyzing existing cyber-attacks that have affected the smart grid, either directly or indirectly.



## **37. Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11576v1) [paper-pdf](http://arxiv.org/pdf/2406.11576v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Junwei Wu, Guoqing Yang, Shaozi Li, Zhiming Luo

**Abstract**: The vulnerability of Deep Neural Networks to adversarial perturbations presents significant security concerns, as the imperceptible perturbations can contaminate the feature space and lead to incorrect predictions. Recent studies have attempted to calibrate contaminated features by either suppressing or over-activating particular channels. Despite these efforts, we claim that adversarial attacks exhibit varying disruption levels across individual channels. Furthermore, we argue that harmonizing feature maps via graph and employing graph convolution can calibrate contaminated features. To this end, we introduce an innovative plug-and-play module called Feature Map-based Reconstructed Graph Convolution (FMR-GC). FMR-GC harmonizes feature maps in the channel dimension to reconstruct the graph, then employs graph convolution to capture neighborhood information, effectively calibrating contaminated features. Extensive experiments have demonstrated the superior performance and scalability of FMR-GC. Moreover, our model can be combined with advanced adversarial training methods to considerably enhance robustness without compromising the model's clean accuracy.



## **38. Do Parameters Reveal More than Loss for Membership Inference?**

cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11544v1) [paper-pdf](http://arxiv.org/pdf/2406.11544v1)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.



## **39. FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11522v1) [paper-pdf](http://arxiv.org/pdf/2406.11522v1)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: Modern machine learning models are sensitive to the manipulation of both the training data (poisoning attacks) and inference data (adversarial examples). Recognizing this issue, the community has developed many empirical defenses against both attacks and, more recently, provable certification methods against inference-time attacks. However, such guarantees are still largely lacking for training-time attacks. In this work, we present FullCert, the first end-to-end certifier with sound, deterministic bounds, which proves robustness against both training-time and inference-time attacks. We first bound all possible perturbations an adversary can make to the training data under the considered threat model. Using these constraints, we bound the perturbations' influence on the model's parameters. Finally, we bound the impact of these parameter changes on the model's prediction, resulting in joint robustness guarantees against poisoning and adversarial examples. To facilitate this novel certification paradigm, we combine our theoretical work with a new open-source library BoundFlow, which enables model training on bounded datasets. We experimentally demonstrate FullCert's feasibility on two different datasets.



## **40. Obfuscating IoT Device Scanning Activity via Adversarial Example Generation**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11515v1) [paper-pdf](http://arxiv.org/pdf/2406.11515v1)

**Authors**: Haocong Li, Yaxin Zhang, Long Cheng, Wenjia Niu, Haining Wang, Qiang Li

**Abstract**: Nowadays, attackers target Internet of Things (IoT) devices for security exploitation, and search engines for devices and services compromise user privacy, including IP addresses, open ports, device types, vendors, and products.Typically, application banners are used to recognize IoT device profiles during network measurement and reconnaissance. In this paper, we propose a novel approach to obfuscating IoT device banners (BANADV) based on adversarial examples. The key idea is to explore the susceptibility of fingerprinting techniques to a slight perturbation of an IoT device banner. By modifying device banners, BANADV disrupts the collection of IoT device profiles. To validate the efficacy of BANADV, we conduct a set of experiments. Our evaluation results show that adversarial examples can spoof state-of-the-art fingerprinting techniques, including learning- and matching-based approaches. We further provide a detailed analysis of the weakness of learning-based/matching-based fingerprints to carefully crafted samples. Overall, the innovations of BANADV lie in three aspects: (1) it utilizes an IoT-related semantic space and a visual similarity space to locate available manipulating perturbations of IoT banners; (2) it achieves at least 80\% success rate for spoofing IoT scanning techniques; and (3) it is the first to utilize adversarial examples of IoT banners in network measurement and reconnaissance.



## **41. Adapters Mixup: Mixing Parameter-Efficient Adapters to Enhance the Adversarial Robustness of Fine-tuned Pre-trained Text Classifiers**

cs.CL

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2401.10111v2) [paper-pdf](http://arxiv.org/pdf/2401.10111v2)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting the training data of pre-trained language models (PLMs) for classification tasks fine-tuned via parameter-efficient fine-tuning methods (PEFT) using both clean and adversarial examples can enhance their robustness under adversarial attacks. However, this adversarial training paradigm often leads to performance degradation on clean inputs and requires frequent re-training on the entire data to account for new, unknown attacks. To overcome these challenges while still harnessing the benefits of adversarial training and the efficiency of PEFT, this work proposes a novel approach, called AdpMixup, that combines two paradigms: (1) fine-tuning through adapters and (2) adversarial augmentation via mixup to dynamically leverage existing knowledge from a set of pre-known attacks for robust inference. Intuitively, AdpMixup fine-tunes PLMs with multiple adapters with both clean and pre-known adversarial examples and intelligently mixes them up in different ratios during prediction. Our experiments show AdpMixup achieves the best trade-off between training efficiency and robustness under both pre-known and unknown attacks, compared to existing baselines on five downstream tasks across six varied black-box attacks and 2 PLMs. All source code will be available.



## **42. ART: Automatic Red-teaming for Text-to-Image Models to Protect Benign Users**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2405.19360v2) [paper-pdf](http://arxiv.org/pdf/2405.19360v2)

**Authors**: Guanlin Li, Kangjie Chen, Shudong Zhang, Jie Zhang, Tianwei Zhang

**Abstract**: Large-scale pre-trained generative models are taking the world by storm, due to their abilities in generating creative content. Meanwhile, safeguards for these generative models are developed, to protect users' rights and safety, most of which are designed for large language models. Existing methods primarily focus on jailbreak and adversarial attacks, which mainly evaluate the model's safety under malicious prompts. Recent work found that manually crafted safe prompts can unintentionally trigger unsafe generations. To further systematically evaluate the safety risks of text-to-image models, we propose a novel Automatic Red-Teaming framework, ART. Our method leverages both vision language model and large language model to establish a connection between unsafe generations and their prompts, thereby more efficiently identifying the model's vulnerabilities. With our comprehensive experiments, we reveal the toxicity of the popular open-source text-to-image models. The experiments also validate the effectiveness, adaptability, and great diversity of ART. Additionally, we introduce three large-scale red-teaming datasets for studying the safety risks associated with text-to-image models. Datasets and models can be found in https://github.com/GuanlinLee/ART.



## **43. $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts**

cs.LG

9 pages, 8 figures, camera ready on ICML2024

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11353v1) [paper-pdf](http://arxiv.org/pdf/2406.11353v1)

**Authors**: Guanjie Chen, Xinyu Zhao, Tianlong Chen, Yu Cheng

**Abstract**: Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoE models sometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose $\texttt{MoE-RBench}$, the first comprehensive assessment of SMoE reliability from three aspects: $\textit{(i)}$ safety and hallucination, $\textit{(ii)}$ resilience to adversarial attacks, and $\textit{(iii)}$ out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoE model to other tasks with higher-generation security, quality, and stability. Codes are available at https://github.com/UNITES-Lab/MoE-RBench



## **44. Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game**

cs.GT

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2305.12872v3) [paper-pdf](http://arxiv.org/pdf/2305.12872v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Ruixiao Xu, Xin Yu, Jiakai Wang, Aishan Liu, Yaodong Yang, Xianglong Liu

**Abstract**: In this study, we explore the robustness of cooperative multi-agent reinforcement learning (c-MARL) against Byzantine failures, where any agent can enact arbitrary, worst-case actions due to malfunction or adversarial attack. To address the uncertainty that any agent can be adversarial, we propose a Bayesian Adversarial Robust Dec-POMDP (BARDec-POMDP) framework, which views Byzantine adversaries as nature-dictated types, represented by a separate transition. This allows agents to learn policies grounded on their posterior beliefs about the type of other agents, fostering collaboration with identified allies and minimizing vulnerability to adversarial manipulation. We define the optimal solution to the BARDec-POMDP as an ex post robust Bayesian Markov perfect equilibrium, which we proof to exist and weakly dominates the equilibrium of previous robust MARL approaches. To realize this equilibrium, we put forward a two-timescale actor-critic algorithm with almost sure convergence under specific conditions. Experimentation on matrix games, level-based foraging and StarCraft II indicate that, even under worst-case perturbations, our method successfully acquires intricate micromanagement skills and adaptively aligns with allies, demonstrating resilience against non-oblivious adversaries, random allies, observation-based attacks, and transfer-based attacks.



## **45. Optimal Attack and Defense for Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2312.00198v2) [paper-pdf](http://arxiv.org/pdf/2312.00198v2)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.



## **46. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

8 pages

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11260v1) [paper-pdf](http://arxiv.org/pdf/2406.11260v1)

**Authors**: Sungwon Park, Sungwon Han, Meeyoung Cha

**Abstract**: The spread of fake news negatively impacts individuals and is regarded as a significant social challenge that needs to be addressed. A number of algorithmic and insightful features have been identified for detecting fake news. However, with the recent LLMs and their advanced generation capabilities, many of the detectable features (e.g., style-conversion attacks) can be altered, making it more challenging to distinguish from real news. This study proposes adversarial style augmentation, AdStyle, to train a fake news detector that remains robust against various style-conversion attacks. Our model's key mechanism is the careful use of LLMs to automatically generate a diverse yet coherent range of style-conversion attack prompts. This improves the generation of prompts that are particularly difficult for the detector to handle. Experiments show that our augmentation strategy improves robustness and detection performance when tested on fake news benchmark datasets.



## **47. The Benefits of Power Regularization in Cooperative Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.11240v1) [paper-pdf](http://arxiv.org/pdf/2406.11240v1)

**Authors**: Michelle Li, Michael Dennis

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (MARL) algorithms, trained only to optimize task reward, can lead to a concentration of power where the failure or adversarial intent of a single agent could decimate the reward of every agent in the system. In the context of teams of people, it is often useful to explicitly consider how power is distributed to ensure no person becomes a single point of failure. Here, we argue that explicitly regularizing the concentration of power in cooperative RL systems can result in systems which are more robust to single agent failure, adversarial attacks, and incentive changes of co-players. To this end, we define a practical pairwise measure of power that captures the ability of any co-player to influence the ego agent's reward, and then propose a power-regularized objective which balances task reward and power concentration. Given this new objective, we show that there always exists an equilibrium where every agent is playing a power-regularized best-response balancing power and task reward. Moreover, we present two algorithms for training agents towards this power-regularized objective: Sample Based Power Regularization (SBPR), which injects adversarial data during training; and Power Regularization via Intrinsic Motivation (PRIM), which adds an intrinsic motivation to regulate power to the training objective. Our experiments demonstrate that both algorithms successfully balance task reward and power, leading to lower power behavior than the baseline of task-only reward and avoid catastrophic events in case an agent in the system goes off-policy.



## **48. ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates**

cs.CR

**SubmitDate**: 2024-06-17    [abs](http://arxiv.org/abs/2406.12935v1) [paper-pdf](http://arxiv.org/pdf/2406.12935v1)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Bill Yuchen Lin, Radha Poovendran

**Abstract**: Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale.   In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research



## **49. Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics**

cs.RO

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2402.10340v4) [paper-pdf](http://arxiv.org/pdf/2402.10340v4)

**Authors**: Xiyang Wu, Souradip Chakraborty, Ruiqi Xian, Jing Liang, Tianrui Guan, Fuxiao Liu, Brian M. Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works focus on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation and navigation. Despite these improvements, analyzing the safety of such systems remains underexplored yet extremely critical. LLMs and VLMs are highly susceptible to adversarial inputs, prompting a significant inquiry into the safety of robotic systems. This concern is important because robotics operate in the physical world where erroneous actions can result in severe consequences. This paper explores this issue thoroughly, presenting a mathematical formulation of potential attacks on LLM/VLM-based robotic systems and offering experimental evidence of the safety challenges. Our empirical findings highlight a significant vulnerability: simple modifications to the input can drastically reduce system effectiveness. Specifically, our results demonstrate an average performance deterioration of 19.4% under minor input prompt modifications and a more alarming 29.1% under slight perceptual changes. These findings underscore the urgent need for robust countermeasures to ensure the safe and reliable deployment of advanced LLM/VLM-based robotic systems.



## **50. garak: A Framework for Security Probing Large Language Models**

cs.CL

https://garak.ai

**SubmitDate**: 2024-06-16    [abs](http://arxiv.org/abs/2406.11036v1) [paper-pdf](http://arxiv.org/pdf/2406.11036v1)

**Authors**: Leon Derczynski, Erick Galinkin, Jeffrey Martin, Subho Majumdar, Nanna Inie

**Abstract**: As Large Language Models (LLMs) are deployed and integrated into thousands of applications, the need for scalable evaluation of how models respond to adversarial attacks grows rapidly. However, LLM security is a moving target: models produce unpredictable output, are constantly updated, and the potential adversary is highly diverse: anyone with access to the internet and a decent command of natural language. Further, what constitutes a security weak in one context may not be an issue in a different context; one-fits-all guardrails remain theoretical. In this paper, we argue that it is time to rethink what constitutes ``LLM security'', and pursue a holistic approach to LLM security evaluation, where exploration and discovery of issues are central. To this end, this paper introduces garak (Generative AI Red-teaming and Assessment Kit), a framework which can be used to discover and identify vulnerabilities in a target LLM or dialog system. garak probes an LLM in a structured fashion to discover potential vulnerabilities. The outputs of the framework describe a target model's weaknesses, contribute to an informed discussion of what composes vulnerabilities in unique contexts, and can inform alignment and policy discussions for LLM deployment.



