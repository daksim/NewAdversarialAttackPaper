# Latest Adversarial Attack Papers
**update at 2023-04-21 11:46:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

cs.CR

15 pages, 13 figures

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2202.03195v5) [paper-pdf](http://arxiv.org/pdf/2202.03195v5)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.



## **2. Certified Adversarial Robustness Within Multiple Perturbation Bounds**

cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10446v1) [paper-pdf](http://arxiv.org/pdf/2304.10446v1)

**Authors**: Soumalya Nandi, Sravanti Addepalli, Harsh Rangwani, R. Venkatesh Babu

**Abstract**: Randomized smoothing (RS) is a well known certified defense against adversarial attacks, which creates a smoothed classifier by predicting the most likely class under random noise perturbations of inputs during inference. While initial work focused on robustness to $\ell_2$ norm perturbations using noise sampled from a Gaussian distribution, subsequent works have shown that different noise distributions can result in robustness to other $\ell_p$ norm bounds as well. In general, a specific noise distribution is optimal for defending against a given $\ell_p$ norm based attack. In this work, we aim to improve the certified adversarial robustness against multiple perturbation bounds simultaneously. Towards this, we firstly present a novel \textit{certification scheme}, that effectively combines the certificates obtained using different noise distributions to obtain optimal results against multiple perturbation bounds. We further propose a novel \textit{training noise distribution} along with a \textit{regularized training scheme} to improve the certification within both $\ell_1$ and $\ell_2$ perturbation norms simultaneously. Contrary to prior works, we compare the certified robustness of different training algorithms across the same natural (clean) accuracy, rather than across fixed noise levels used for training and certification. We also empirically invalidate the argument that training and certifying the classifier with the same amount of noise gives the best results. The proposed approach achieves improvements on the ACR (Average Certified Radius) metric across both $\ell_1$ and $\ell_2$ perturbation bounds.



## **3. An Analysis of the Completion Time of the BB84 Protocol**

cs.PF

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10218v1) [paper-pdf](http://arxiv.org/pdf/2304.10218v1)

**Authors**: Sounak Kar, Jean-Yves Le Boudec

**Abstract**: The BB84 QKD protocol is based on the idea that the sender and the receiver can reconcile a certain fraction of the teleported qubits to detect eavesdropping or noise and decode the rest to use as a private key. Under the present hardware infrastructure, decoherence of quantum states poses a significant challenge to performing perfect or efficient teleportation, meaning that a teleportation-based protocol must be run multiple times to observe success. Thus, performance analyses of such protocols usually consider the completion time, i.e., the time until success, rather than the duration of a single attempt. Moreover, due to decoherence, the success of an attempt is in general dependent on the duration of individual phases of that attempt, as quantum states must wait in memory while the success or failure of a generation phase is communicated to the relevant parties. In this work, we do a performance analysis of the completion time of the BB84 protocol in a setting where the sender and the receiver are connected via a single quantum repeater and the only quantum channel between them does not see any adversarial attack. Assuming certain distributional forms for the generation and communication phases of teleportation, we provide a method to compute the MGF of the completion time and subsequently derive an estimate of the CDF and a bound on the tail probability. This result helps us gauge the (tail) behaviour of the completion time in terms of the parameters characterising the elementary phases of teleportation, without having to run the protocol multiple times. We also provide an efficient simulation scheme to generate the completion time, which relies on expressing the completion time in terms of aggregated teleportation times. We numerically compare our approach with a full-scale simulation and observe good agreement between them.



## **4. Quantum-secure message authentication via blind-unforgeability**

quant-ph

37 pages, v4: Erratum added. We removed a result that had an error in  its proof

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/1803.03761v4) [paper-pdf](http://arxiv.org/pdf/1803.03761v4)

**Authors**: Gorjan Alagic, Christian Majenz, Alexander Russell, Fang Song

**Abstract**: Formulating and designing authentication of classical messages in the presence of adversaries with quantum query access has been a longstanding challenge, as the familiar classical notions of unforgeability do not directly translate into meaningful notions in the quantum setting. A particular difficulty is how to fairly capture the notion of "predicting an unqueried value" when the adversary can query in quantum superposition.   We propose a natural definition of unforgeability against quantum adversaries called blind unforgeability. This notion defines a function to be predictable if there exists an adversary who can use "partially blinded" oracle access to predict values in the blinded region. We support the proposal with a number of technical results. We begin by establishing that the notion coincides with EUF-CMA in the classical setting and go on to demonstrate that the notion is satisfied by a number of simple guiding examples, such as random functions and quantum-query-secure pseudorandom functions. We then show the suitability of blind unforgeability for supporting canonical constructions and reductions. We prove that the "hash-and-MAC" paradigm and the Lamport one-time digital signature scheme are indeed unforgeable according to the definition. To support our analysis, we additionally define and study a new variety of quantum-secure hash functions called Bernoulli-preserving.   Finally, we demonstrate that blind unforgeability is stronger than a previous definition of Boneh and Zhandry [EUROCRYPT '13, CRYPTO '13] in the sense that we can construct an explicit function family which is forgeable by an attack that is recognized by blind-unforgeability, yet satisfies the definition by Boneh and Zhandry.



## **5. Diversifying the High-level Features for better Adversarial Transferability**

cs.CV

15 pages

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10136v1) [paper-pdf](http://arxiv.org/pdf/2304.10136v1)

**Authors**: Zhiyuan Wang, Zeliang Zhang, Siyuan Liang, Xiaosen Wang

**Abstract**: Given the great threat of adversarial attacks against Deep Neural Networks (DNNs), numerous works have been proposed to boost transferability to attack real-world applications. However, existing attacks often utilize advanced gradient calculation or input transformation but ignore the white-box model. Inspired by the fact that DNNs are over-parameterized for superior performance, we propose diversifying the high-level features (DHF) for more transferable adversarial examples. In particular, DHF perturbs the high-level features by randomly transforming the high-level features and mixing them with the feature of benign samples when calculating the gradient at each iteration. Due to the redundancy of parameters, such transformation does not affect the classification performance but helps identify the invariant features across different models, leading to much better transferability. Empirical evaluations on ImageNet dataset show that DHF could effectively improve the transferability of existing momentum-based attacks. Incorporated into the input transformation-based attacks, DHF generates more transferable adversarial examples and outperforms the baselines with a clear margin when attacking several defense models, showing its generalization to various attacks and high effectiveness for boosting transferability.



## **6. Towards the Universal Defense for Query-Based Audio Adversarial Attacks**

eess.AS

Submitted to Cybersecurity journal

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10088v1) [paper-pdf](http://arxiv.org/pdf/2304.10088v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: Recently, studies show that deep learning-based automatic speech recognition (ASR) systems are vulnerable to adversarial examples (AEs), which add a small amount of noise to the original audio examples. These AE attacks pose new challenges to deep learning security and have raised significant concerns about deploying ASR systems and devices. The existing defense methods are either limited in application or only defend on results, but not on process. In this work, we propose a novel method to infer the adversary intent and discover audio adversarial examples based on the AEs generation process. The insight of this method is based on the observation: many existing audio AE attacks utilize query-based methods, which means the adversary must send continuous and similar queries to target ASR models during the audio AE generation process. Inspired by this observation, We propose a memory mechanism by adopting audio fingerprint technology to analyze the similarity of the current query with a certain length of memory query. Thus, we can identify when a sequence of queries appears to be suspectable to generate audio AEs. Through extensive evaluation on four state-of-the-art audio AE attacks, we demonstrate that on average our defense identify the adversary intent with over 90% accuracy. With careful regard for robustness evaluations, we also analyze our proposed defense and its strength to withstand two adaptive attacks. Finally, our scheme is available out-of-the-box and directly compatible with any ensemble of ASR defense models to uncover audio AE attacks effectively without model retraining.



## **7. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

cs.SE

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2206.07813v3) [paper-pdf](http://arxiv.org/pdf/2206.07813v3)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.



## **8. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

stat.ML

This paper was published in Pattern Recognition

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2009.04709v5) [paper-pdf](http://arxiv.org/pdf/2009.04709v5)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.



## **9. Jedi: Entropy-based Localization and Removal of Adversarial Patches**

cs.CR

9 pages, 11 figures. To appear in CVPR 2023

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10029v1) [paper-pdf](http://arxiv.org/pdf/2304.10029v1)

**Authors**: Bilel Tarchoun, Anouar Ben Khalifa, Mohamed Ali Mahjoub, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Real-world adversarial physical patches were shown to be successful in compromising state-of-the-art models in a variety of computer vision applications. Existing defenses that are based on either input gradient or features analysis have been compromised by recent GAN-based attacks that generate naturalistic patches. In this paper, we propose Jedi, a new defense against adversarial patches that is resilient to realistic patch attacks. Jedi tackles the patch localization problem from an information theory perspective; leverages two new ideas: (1) it improves the identification of potential patch regions using entropy analysis: we show that the entropy of adversarial patches is high, even in naturalistic patches; and (2) it improves the localization of adversarial patches, using an autoencoder that is able to complete patch regions from high entropy kernels. Jedi achieves high-precision adversarial patch localization, which we show is critical to successfully repair the images. Since Jedi relies on an input entropy analysis, it is model-agnostic, and can be applied on pre-trained off-the-shelf models without changes to the training or inference of the protected models. Jedi detects on average 90% of adversarial patches across different benchmarks and recovers up to 94% of successful patch attacks (Compared to 75% and 65% for LGS and Jujutsu, respectively).



## **10. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

cs.LG

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09875v1) [paper-pdf](http://arxiv.org/pdf/2304.09875v1)

**Authors**: Li Zaitang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.



## **11. Experimental Certification of Quantum Transmission via Bell's Theorem**

quant-ph

34 pages, 14 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09605v1) [paper-pdf](http://arxiv.org/pdf/2304.09605v1)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all implementations of quantum information protocols. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted state itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.



## **12. Masked Language Model Based Textual Adversarial Example Detection**

cs.CR

13 pages,3 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.08767v2) [paper-pdf](http://arxiv.org/pdf/2304.08767v2)

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Qi Zhong, Xufei Zheng, Yanjun Zhang, Shengshan Hu, Leo Yu Zhang

**Abstract**: Adversarial attacks are a serious threat to the reliable deployment of machine learning models in safety-critical applications. They can misguide current models to predict incorrectly by slightly modifying the inputs. Recently, substantial work has shown that adversarial examples tend to deviate from the underlying data manifold of normal examples, whereas pre-trained masked language models can fit the manifold of normal NLP data. To explore how to use the masked language model in adversarial detection, we propose a novel textual adversarial example detection method, namely Masked Language Model-based Detection (MLMD), which can produce clearly distinguishable signals between normal examples and adversarial examples by exploring the changes in manifolds induced by the masked language model. MLMD features a plug and play usage (i.e., no need to retrain the victim model) for adversarial defense and it is agnostic to classification tasks, victim model's architectures, and to-be-defended attack methods. We evaluate MLMD on various benchmark textual datasets, widely studied machine learning models, and state-of-the-art (SOTA) adversarial attacks (in total $3*4*4 = 48$ settings). Experimental results show that MLMD can achieve strong performance, with detection accuracy up to 0.984, 0.967, and 0.901 on AG-NEWS, IMDB, and SST-2 datasets, respectively. Additionally, MLMD is superior, or at least comparable to, the SOTA detection defenses in detection accuracy and F1 score. Among many defenses based on the off-manifold assumption of adversarial examples, this work offers a new angle for capturing the manifold change. The code for this work is openly accessible at \url{https://github.com/mlmddetection/MLMDdetection}.



## **13. Understanding Overfitting in Adversarial Training via Kernel Regression**

stat.ML

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.06326v2) [paper-pdf](http://arxiv.org/pdf/2304.06326v2)

**Authors**: Teng Zhang, Kang Li

**Abstract**: Adversarial training and data augmentation with noise are widely adopted techniques to enhance the performance of neural networks. This paper investigates adversarial training and data augmentation with noise in the context of regularized regression in a reproducing kernel Hilbert space (RKHS). We establish the limiting formula for these techniques as the attack and noise size, as well as the regularization parameter, tend to zero. Based on this limiting formula, we analyze specific scenarios and demonstrate that, without appropriate regularization, these two methods may have larger generalization error and Lipschitz constant than standard kernel regression. However, by selecting the appropriate regularization parameter, these two methods can outperform standard kernel regression and achieve smaller generalization error and Lipschitz constant. These findings support the empirical observations that adversarial training can lead to overfitting, and appropriate regularization methods, such as early stopping, can alleviate this issue.



## **14. Secure Split Learning against Property Inference, Data Reconstruction, and Feature Space Hijacking Attacks**

cs.LG

23 pages

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09515v1) [paper-pdf](http://arxiv.org/pdf/2304.09515v1)

**Authors**: Yunlong Mao, Zexi Xin, Zhenyu Li, Jue Hong, Qingyou Yang, Sheng Zhong

**Abstract**: Split learning of deep neural networks (SplitNN) has provided a promising solution to learning jointly for the mutual interest of a guest and a host, which may come from different backgrounds, holding features partitioned vertically. However, SplitNN creates a new attack surface for the adversarial participant, holding back its practical use in the real world. By investigating the adversarial effects of highly threatening attacks, including property inference, data reconstruction, and feature hijacking attacks, we identify the underlying vulnerability of SplitNN and propose a countermeasure. To prevent potential threats and ensure the learning guarantees of SplitNN, we design a privacy-preserving tunnel for information exchange between the guest and the host. The intuition is to perturb the propagation of knowledge in each direction with a controllable unified solution. To this end, we propose a new activation function named R3eLU, transferring private smashed data and partial loss into randomized responses in forward and backward propagations, respectively. We give the first attempt to secure split learning against three threatening attacks and present a fine-grained privacy budget allocation scheme. The analysis proves that our privacy-preserving SplitNN solution provides a tight privacy budget, while the experimental results show that our solution performs better than existing solutions in most cases and achieves a good tradeoff between defense and model usability.



## **15. Maybenot: A Framework for Traffic Analysis Defenses**

cs.CR

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09510v1) [paper-pdf](http://arxiv.org/pdf/2304.09510v1)

**Authors**: Tobias Pulls

**Abstract**: End-to-end encryption is a powerful tool for protecting the privacy of Internet users. Together with the increasing use of technologies such as Tor, VPNs, and encrypted messaging, it is becoming increasingly difficult for network adversaries to monitor and censor Internet traffic. One remaining avenue for adversaries is traffic analysis: the analysis of patterns in encrypted traffic to infer information about the users and their activities. Recent improvements using deep learning have made traffic analysis attacks more effective than ever before.   We present Maybenot, a framework for traffic analysis defenses. Maybenot is designed to be easy to use and integrate into existing end-to-end encrypted protocols. It is implemented in the Rust programming language as a crate (library), together with a simulator to further the development of defenses. Defenses in Maybenot are expressed as probabilistic state machines that schedule actions to inject padding or block outgoing traffic. Maybenot is an evolution from the Tor Circuit Padding Framework by Perry and Kadianakis, designed to support a wide range of protocols and use cases.



## **16. Wavelets Beat Monkeys at Adversarial Robustness**

cs.LG

Machine Learning and the Physical Sciences Workshop, NeurIPS 2022

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09403v1) [paper-pdf](http://arxiv.org/pdf/2304.09403v1)

**Authors**: Jingtong Su, Julia Kempe

**Abstract**: Research on improving the robustness of neural networks to adversarial noise - imperceptible malicious perturbations of the data - has received significant attention. The currently uncontested state-of-the-art defense to obtain robust deep neural networks is Adversarial Training (AT), but it consumes significantly more resources compared to standard training and trades off accuracy for robustness. An inspiring recent work [Dapello et al.] aims to bring neurobiological tools to the question: How can we develop Neural Nets that robustly generalize like human vision? [Dapello et al.] design a network structure with a neural hidden first layer that mimics the primate primary visual cortex (V1), followed by a back-end structure adapted from current CNN vision models. It seems to achieve non-trivial adversarial robustness on standard vision benchmarks when tested on small perturbations. Here we revisit this biologically inspired work, and ask whether a principled parameter-free representation with inspiration from physics is able to achieve the same goal. We discover that the wavelet scattering transform can replace the complex V1-cortex and simple uniform Gaussian noise can take the role of neural stochasticity, to achieve adversarial robustness. In extensive experiments on the CIFAR-10 benchmark with adaptive adversarial attacks we show that: 1) Robustness of VOneBlock architectures is relatively weak (though non-zero) when the strength of the adversarial attack radius is set to commonly used benchmarks. 2) Replacing the front-end VOneBlock by an off-the-shelf parameter-free Scatternet followed by simple uniform Gaussian noise can achieve much more substantial adversarial robustness without adversarial training. Our work shows how physically inspired structures yield new insights into robustness that were previously only thought possible by meticulously mimicking the human cortex.



## **17. CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models**

cs.CL

AAAI Conference on Artificial Intelligence (AAAI) 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2206.00052v3) [paper-pdf](http://arxiv.org/pdf/2206.00052v3)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstract**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.



## **18. Analyzing Activity and Suspension Patterns of Twitter Bots Attacking Turkish Twitter Trends by a Longitudinal Dataset**

cs.SI

Accepted to Cyber Social Threats (CySoc) 2023 colocated with  WebConf23

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.07907v2) [paper-pdf](http://arxiv.org/pdf/2304.07907v2)

**Authors**: Tuğrulcan Elmas

**Abstract**: Twitter bots amplify target content in a coordinated manner to make them appear popular, which is an astroturfing attack. Such attacks promote certain keywords to push them to Twitter trends to make them visible to a broader audience. Past work on such fake trends revealed a new astroturfing attack named ephemeral astroturfing that employs a very unique bot behavior in which bots post and delete generated tweets in a coordinated manner. As such, it is easy to mass-annotate such bots reliably, making them a convenient source of ground truth for bot research. In this paper, we detect and disclose over 212,000 such bots targeting Turkish trends, which we name astrobots. We also analyze their activity and suspension patterns. We found that Twitter purged those bots en-masse 6 times since June 2018. However, the adversaries reacted quickly and deployed new bots that were created years ago. We also found that many such bots do not post tweets apart from promoting fake trends, which makes it challenging for bot detection methods to detect them. Our work provides insights into platforms' content moderation practices and bot detection research. The dataset is publicly available at https://github.com/tugrulz/EphemeralAstroturfing.



## **19. An Analysis of Robustness of Non-Lipschitz Networks**

cs.LG

To appear in Journal of Machine Learning Research (JMLR)

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2010.06154v4) [paper-pdf](http://arxiv.org/pdf/2010.06154v4)

**Authors**: Maria-Florina Balcan, Avrim Blum, Dravyansh Sharma, Hongyang Zhang

**Abstract**: Despite significant advances, deep networks remain highly susceptible to adversarial attack. One fundamental challenge is that small input perturbations can often produce large movements in the network's final-layer feature space. In this paper, we define an attack model that abstracts this challenge, to help understand its intrinsic properties. In our model, the adversary may move data an arbitrary distance in feature space but only in random low-dimensional subspaces. We prove such adversaries can be quite powerful: defeating any algorithm that must classify any input it is given. However, by allowing the algorithm to abstain on unusual inputs, we show such adversaries can be overcome when classes are reasonably well-separated in feature space. We further provide strong theoretical guarantees for setting algorithm parameters to optimize over accuracy-abstention trade-offs using data-driven methods. Our results provide new robustness guarantees for nearest-neighbor style algorithms, and also have application to contrastive learning, where we empirically demonstrate the ability of such algorithms to obtain high robust accuracy with low abstention rates. Our model is also motivated by strategic classification, where entities being classified aim to manipulate their observable features to produce a preferred classification, and we provide new insights into that area as well.



## **20. BadVFL: Backdoor Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.08847v1) [paper-pdf](http://arxiv.org/pdf/2304.08847v1)

**Authors**: Mohammad Naseri, Yufei Han, Emiliano De Cristofaro

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively train a machine learning model without sharing their data; rather, they train their own model locally and send updates to a central server for aggregation. Depending on how the data is distributed among the participants, FL can be classified into Horizontal (HFL) and Vertical (VFL). In VFL, the participants share the same set of training instances but only host a different and non-overlapping subset of the whole feature space. Whereas in HFL, each participant shares the same set of features while the training set is split into locally owned training data subsets.   VFL is increasingly used in applications like financial fraud detection; nonetheless, very little work has analyzed its security. In this paper, we focus on robustness in VFL, in particular, on backdoor attacks, whereby an adversary attempts to manipulate the aggregate model during the training process to trigger misclassifications. Performing backdoor attacks in VFL is more challenging than in HFL, as the adversary i) does not have access to the labels during training and ii) cannot change the labels as she only has access to the feature embeddings. We present a first-of-its-kind clean-label backdoor attack in VFL, which consists of two phases: a label inference and a backdoor phase. We demonstrate the effectiveness of the attack on three different datasets, investigate the factors involved in its success, and discuss countermeasures to mitigate its impact.



## **21. Towards the Transferable Audio Adversarial Attack via Ensemble Methods**

cs.CR

Submitted to Cybersecurity journal 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.08811v1) [paper-pdf](http://arxiv.org/pdf/2304.08811v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: In recent years, deep learning (DL) models have achieved significant progress in many domains, such as autonomous driving, facial recognition, and speech recognition. However, the vulnerability of deep learning models to adversarial attacks has raised serious concerns in the community because of their insufficient robustness and generalization. Also, transferable attacks have become a prominent method for black-box attacks. In this work, we explore the potential factors that impact adversarial examples (AEs) transferability in DL-based speech recognition. We also discuss the vulnerability of different DL systems and the irregular nature of decision boundaries. Our results show a remarkable difference in the transferability of AEs between speech and images, with the data relevance being low in images but opposite in speech recognition. Motivated by dropout-based ensemble approaches, we propose random gradient ensembles and dynamic gradient-weighted ensembles, and we evaluate the impact of ensembles on the transferability of AEs. The results show that the AEs created by both approaches are valid for transfer to the black box API.



## **22. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022, Best Paper Nomination

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2209.06506v2) [paper-pdf](http://arxiv.org/pdf/2209.06506v2)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstract**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.



## **23. A Survey of Adversarial Defences and Robustness in NLP**

cs.CL

Accepted for publication at ACM Computing Surveys

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2203.06414v4) [paper-pdf](http://arxiv.org/pdf/2203.06414v4)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstract**: In the past few years, it has become increasingly evident that deep neural networks are not resilient enough to withstand adversarial perturbations in input data, leaving them vulnerable to attack. Various authors have proposed strong adversarial attacks for computer vision and Natural Language Processing (NLP) tasks. As a response, many defense mechanisms have also been proposed to prevent these networks from failing. The significance of defending neural networks against adversarial attacks lies in ensuring that the model's predictions remain unchanged even if the input data is perturbed. Several methods for adversarial defense in NLP have been proposed, catering to different NLP tasks such as text classification, named entity recognition, and natural language inference. Some of these methods not only defend neural networks against adversarial attacks but also act as a regularization mechanism during training, saving the model from overfitting. This survey aims to review the various methods proposed for adversarial defenses in NLP over the past few years by introducing a novel taxonomy. The survey also highlights the fragility of advanced deep neural networks in NLP and the challenges involved in defending them.



## **24. Binarized ResNet: Enabling Robust Automatic Modulation Classification at the resource-constrained Edge**

cs.IT

This version has a total of 8 figures and 3 tables. It has extra  content on the adversarial robustness of the proposed method that was not  present in the previous submission. Also one more ensemble method called  RBLResNet-MCK is proposed to improve the performance further

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2110.14357v2) [paper-pdf](http://arxiv.org/pdf/2110.14357v2)

**Authors**: Deepsayan Sadhukhan, Nitin Priyadarshini Shankar, Nancy Nayak, Thulasi Tholeti, Sheetal Kalyani

**Abstract**: Recently, deep neural networks (DNNs) have been used extensively for automatic modulation classification (AMC), and the results have been quite promising. However, DNNs have high memory and computation requirements making them impractical for edge networks where the devices are resource-constrained. They are also vulnerable to adversarial attacks, which is a significant security concern. This work proposes a rotated binary large ResNet (RBLResNet) for AMC that can be deployed at the edge network because of low memory and computational complexity. The performance gap between the RBLResNet and existing architectures with floating-point weights and activations can be closed by two proposed ensemble methods: (i) multilevel classification (MC), and (ii) bagging multiple RBLResNets while retaining low memory and computational power. The MC method achieves an accuracy of $93.39\%$ at $10$dB over all the $24$ modulation classes of the Deepsig dataset. This performance is comparable to state-of-the-art (SOTA) performances, with $4.75$ times lower memory and $1214$ times lower computation. Furthermore, RBLResNet also has high adversarial robustness compared to existing DNN models. The proposed MC method with RBLResNets has an adversarial accuracy of $87.25\%$ over a wide range of SNRs, surpassing the robustness of all existing SOTA methods to the best of our knowledge. Properties such as low memory, low computation, and the highest adversarial robustness make it a better choice for robust AMC in low-power edge devices.



## **25. Employing Deep Ensemble Learning for Improving the Security of Computer Networks against Adversarial Attacks**

cs.CR

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2209.12195v2) [paper-pdf](http://arxiv.org/pdf/2209.12195v2)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Erkay Savas, Mauro Conti, Yassine Mekdad

**Abstract**: In the past few years, Convolutional Neural Networks (CNN) have demonstrated promising performance in various real-world cybersecurity applications, such as network and multimedia security. However, the underlying fragility of CNN structures poses major security problems, making them inappropriate for use in security-oriented applications including such computer networks. Protecting these architectures from adversarial attacks necessitates using security-wise architectures that are challenging to attack.   In this study, we present a novel architecture based on an ensemble classifier that combines the enhanced security of 1-Class classification (known as 1C) with the high performance of conventional 2-Class classification (known as 2C) in the absence of attacks.Our architecture is referred to as the 1.5-Class (SPRITZ-1.5C) classifier and constructed using a final dense classifier, one 2C classifier (i.e., CNNs), and two parallel 1C classifiers (i.e., auto-encoders). In our experiments, we evaluated the robustness of our proposed architecture by considering eight possible adversarial attacks in various scenarios. We performed these attacks on the 2C and SPRITZ-1.5C architectures separately. The experimental results of our study showed that the Attack Success Rate (ASR) of the I-FGSM attack against a 2C classifier trained with the N-BaIoT dataset is 0.9900. In contrast, the ASR is 0.0000 for the SPRITZ-1.5C classifier.



## **26. RNN-Guard: Certified Robustness Against Multi-frame Attacks for Recurrent Neural Networks**

cs.LG

13 pages, 7 figures, 6 tables

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2304.07980v1) [paper-pdf](http://arxiv.org/pdf/2304.07980v1)

**Authors**: Yunruo Zhang, Tianyu Du, Shouling Ji, Peng Tang, Shanqing Guo

**Abstract**: It is well-known that recurrent neural networks (RNNs), although widely used, are vulnerable to adversarial attacks including one-frame attacks and multi-frame attacks. Though a few certified defenses exist to provide guaranteed robustness against one-frame attacks, we prove that defending against multi-frame attacks remains a challenging problem due to their enormous perturbation space. In this paper, we propose the first certified defense against multi-frame attacks for RNNs called RNN-Guard. To address the above challenge, we adopt the perturb-all-frame strategy to construct perturbation spaces consistent with those in multi-frame attacks. However, the perturb-all-frame strategy causes a precision issue in linear relaxations. To address this issue, we introduce a novel abstract domain called InterZono and design tighter relaxations. We prove that InterZono is more precise than Zonotope yet carries the same time complexity. Experimental evaluations across various datasets and model structures show that the certified robust accuracy calculated by RNN-Guard with InterZono is up to 2.18 times higher than that with Zonotope. In addition, we extend RNN-Guard as the first certified training method against multi-frame attacks to directly enhance RNNs' robustness. The results show that the certified robust accuracy of models trained with RNN-Guard against multi-frame attacks is 15.47 to 67.65 percentage points higher than those with other training methods.



## **27. A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness**

cs.SD

**SubmitDate**: 2023-04-16    [abs](http://arxiv.org/abs/2212.09006v2) [paper-pdf](http://arxiv.org/pdf/2212.09006v2)

**Authors**: Tiantian Feng, Rajat Hebbar, Nicholas Mehlman, Xuan Shi, Aditya Kommineni, and Shrikanth Narayanan

**Abstract**: Speech-centric machine learning systems have revolutionized many leading domains ranging from transportation and healthcare to education and defense, profoundly changing how people live, work, and interact with each other. However, recent studies have demonstrated that many speech-centric ML systems may need to be considered more trustworthy for broader deployment. Specifically, concerns over privacy breaches, discriminating performance, and vulnerability to adversarial attacks have all been discovered in ML research fields. In order to address the above challenges and risks, a significant number of efforts have been made to ensure these ML systems are trustworthy, especially private, safe, and fair. In this paper, we conduct the first comprehensive survey on speech-centric trustworthy ML topics related to privacy, safety, and fairness. In addition to serving as a summary report for the research community, we point out several promising future research directions to inspire the researchers who wish to explore further in this area.



## **28. Visual Prompting for Adversarial Robustness**

cs.CV

ICASSP 2023

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2210.06284v3) [paper-pdf](http://arxiv.org/pdf/2210.06284v3)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.



## **29. XploreNAS: Explore Adversarially Robust & Hardware-efficient Neural Architectures for Non-ideal Xbars**

cs.LG

Accepted to ACM Transactions on Embedded Computing Systems in April  2023

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2302.07769v2) [paper-pdf](http://arxiv.org/pdf/2302.07769v2)

**Authors**: Abhiroop Bhattacharjee, Abhishek Moitra, Priyadarshini Panda

**Abstract**: Compute In-Memory platforms such as memristive crossbars are gaining focus as they facilitate acceleration of Deep Neural Networks (DNNs) with high area and compute-efficiencies. However, the intrinsic non-idealities associated with the analog nature of computing in crossbars limits the performance of the deployed DNNs. Furthermore, DNNs are shown to be vulnerable to adversarial attacks leading to severe security threats in their large-scale deployment. Thus, finding adversarially robust DNN architectures for non-ideal crossbars is critical to the safe and secure deployment of DNNs on the edge. This work proposes a two-phase algorithm-hardware co-optimization approach called XploreNAS that searches for hardware-efficient & adversarially robust neural architectures for non-ideal crossbar platforms. We use the one-shot Neural Architecture Search (NAS) approach to train a large Supernet with crossbar-awareness and sample adversarially robust Subnets therefrom, maintaining competitive hardware-efficiency. Our experiments on crossbars with benchmark datasets (SVHN, CIFAR10 & CIFAR100) show upto ~8-16% improvement in the adversarial robustness of the searched Subnets against a baseline ResNet-18 model subjected to crossbar-aware adversarial training. We benchmark our robust Subnets for Energy-Delay-Area-Products (EDAPs) using the Neurosim tool and find that with additional hardware-efficiency driven optimizations, the Subnets attain ~1.5-1.6x lower EDAPs than ResNet-18 baseline.



## **30. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

cs.CV

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2211.01671v4) [paper-pdf](http://arxiv.org/pdf/2211.01671v4)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.



## **31. Combining Generators of Adversarial Malware Examples to Increase Evasion Rate**

cs.CR

9 pages, 5 figures, 2 tables. Under review

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.07360v1) [paper-pdf](http://arxiv.org/pdf/2304.07360v1)

**Authors**: Matouš Kozák, Martin Jureček

**Abstract**: Antivirus developers are increasingly embracing machine learning as a key component of malware defense. While machine learning achieves cutting-edge outcomes in many fields, it also has weaknesses that are exploited by several adversarial attack techniques. Many authors have presented both white-box and black-box generators of adversarial malware examples capable of bypassing malware detectors with varying success. We propose to combine contemporary generators in order to increase their potential. Combining different generators can create more sophisticated adversarial examples that are more likely to evade anti-malware tools. We demonstrated this technique on five well-known generators and recorded promising results. The best-performing combination of AMG-random and MAB-Malware generators achieved an average evasion rate of 15.9% against top-tier antivirus products. This represents an average improvement of more than 36% and 627% over using only the AMG-random and MAB-Malware generators, respectively. The generator that benefited the most from having another generator follow its procedure was the FGSM injection attack, which improved the evasion rate on average between 91.97% and 1,304.73%, depending on the second generator used. These results demonstrate that combining different generators can significantly improve their effectiveness against leading antivirus programs.



## **32. Pool Inference Attacks on Local Differential Privacy: Quantifying the Privacy Guarantees of Apple's Count Mean Sketch in Practice**

cs.CR

Published at USENIX Security 2022. This is the full version, please  cite the USENIX version (see journal reference field)

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.07134v1) [paper-pdf](http://arxiv.org/pdf/2304.07134v1)

**Authors**: Andrea Gadotti, Florimond Houssiau, Meenatchi Sundaram Muthu Selva Annamalai, Yves-Alexandre de Montjoye

**Abstract**: Behavioral data generated by users' devices, ranging from emoji use to pages visited, are collected at scale to improve apps and services. These data, however, contain fine-grained records and can reveal sensitive information about individual users. Local differential privacy has been used by companies as a solution to collect data from users while preserving privacy. We here first introduce pool inference attacks, where an adversary has access to a user's obfuscated data, defines pools of objects, and exploits the user's polarized behavior in multiple data collections to infer the user's preferred pool. Second, we instantiate this attack against Count Mean Sketch, a local differential privacy mechanism proposed by Apple and deployed in iOS and Mac OS devices, using a Bayesian model. Using Apple's parameters for the privacy loss $\varepsilon$, we then consider two specific attacks: one in the emojis setting -- where an adversary aims at inferring a user's preferred skin tone for emojis -- and one against visited websites -- where an adversary wants to learn the political orientation of a user from the news websites they visit. In both cases, we show the attack to be much more effective than a random guess when the adversary collects enough data. We find that users with high polarization and relevant interest are significantly more vulnerable, and we show that our attack is well-calibrated, allowing the adversary to target such vulnerable users. We finally validate our results for the emojis setting using user data from Twitter. Taken together, our results show that pool inference attacks are a concern for data protected by local differential privacy mechanisms with a large $\varepsilon$, emphasizing the need for additional technical safeguards and the need for more research on how to apply local differential privacy for multiple collections.



## **33. Interpretability is a Kind of Safety: An Interpreter-based Ensemble for Adversary Defense**

cs.LG

10 pages, accepted to KDD'20

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.06919v1) [paper-pdf](http://arxiv.org/pdf/2304.06919v1)

**Authors**: Jingyuan Wang, Yufan Wu, Mingxuan Li, Xin Lin, Junjie Wu, Chao Li

**Abstract**: While having achieved great success in rich real-life applications, deep neural network (DNN) models have long been criticized for their vulnerability to adversarial attacks. Tremendous research efforts have been dedicated to mitigating the threats of adversarial attacks, but the essential trait of adversarial examples is not yet clear, and most existing methods are yet vulnerable to hybrid attacks and suffer from counterattacks. In light of this, in this paper, we first reveal a gradient-based correlation between sensitivity analysis-based DNN interpreters and the generation process of adversarial examples, which indicates the Achilles's heel of adversarial attacks and sheds light on linking together the two long-standing challenges of DNN: fragility and unexplainability. We then propose an interpreter-based ensemble framework called X-Ensemble for robust adversary defense. X-Ensemble adopts a novel detection-rectification process and features in building multiple sub-detectors and a rectifier upon various types of interpretation information toward target classifiers. Moreover, X-Ensemble employs the Random Forests (RF) model to combine sub-detectors into an ensemble detector for adversarial hybrid attacks defense. The non-differentiable property of RF further makes it a precious choice against the counterattack of adversaries. Extensive experiments under various types of state-of-the-art attacks and diverse attack scenarios demonstrate the advantages of X-Ensemble to competitive baseline methods.



## **34. Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

cs.LG

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2207.09572v3) [paper-pdf](http://arxiv.org/pdf/2207.09572v3)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstract**: This work studies the threats of adversarial attack on multivariate probabilistic forecasting models and viable defense mechanisms. Our studies discover a new attack pattern that negatively impact the forecasting of a target time series via making strategic, sparse (imperceptible) modifications to the past observations of a small number of other time series. To mitigate the impact of such attack, we have developed two defense strategies. First, we extend a previously developed randomized smoothing technique in classification to multivariate forecasting scenarios. Second, we develop an adversarial training algorithm that learns to create adversarial examples and at the same time optimizes the forecasting model to improve its robustness against such adversarial simulation. Extensive experiments on real-world datasets confirm that our attack schemes are powerful and our defense algorithms are more effective compared with baseline defense mechanisms.



## **35. Generating Adversarial Examples with Better Transferability via Masking Unimportant Parameters of Surrogate Model**

cs.LG

Accepted at 2023 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.06908v1) [paper-pdf](http://arxiv.org/pdf/2304.06908v1)

**Authors**: Dingcheng Yang, Wenjian Yu, Zihao Xiao, Jiaqi Luo

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples. Moreover, the transferability of the adversarial examples has received broad attention in recent years, which means that adversarial examples crafted by a surrogate model can also attack unknown models. This phenomenon gave birth to the transfer-based adversarial attacks, which aim to improve the transferability of the generated adversarial examples. In this paper, we propose to improve the transferability of adversarial examples in the transfer-based attack via masking unimportant parameters (MUP). The key idea in MUP is to refine the pretrained surrogate models to boost the transfer-based attack. Based on this idea, a Taylor expansion-based metric is used to evaluate the parameter importance score and the unimportant parameters are masked during the generation of adversarial examples. This process is simple, yet can be naturally combined with various existing gradient-based optimizers for generating adversarial examples, thus further improving the transferability of the generated adversarial examples. Extensive experiments are conducted to validate the effectiveness of the proposed MUP-based methods.



## **36. Don't Knock! Rowhammer at the Backdoor of DNN Models**

cs.LG

2023 53rd Annual IEEE/IFIP International Conference on Dependable  Systems and Networks (DSN)

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2110.07683v3) [paper-pdf](http://arxiv.org/pdf/2110.07683v3)

**Authors**: M. Caner Tol, Saad Islam, Andrew J. Adiletta, Berk Sunar, Ziming Zhang

**Abstract**: State-of-the-art deep neural networks (DNNs) have been proven to be vulnerable to adversarial manipulation and backdoor attacks. Backdoored models deviate from expected behavior on inputs with predefined triggers while retaining performance on clean data. Recent works focus on software simulation of backdoor injection during the inference phase by modifying network weights, which we find often unrealistic in practice due to restrictions in hardware.   In contrast, in this work for the first time, we present an end-to-end backdoor injection attack realized on actual hardware on a classifier model using Rowhammer as the fault injection method. To this end, we first investigate the viability of backdoor injection attacks in real-life deployments of DNNs on hardware and address such practical issues in hardware implementation from a novel optimization perspective. We are motivated by the fact that vulnerable memory locations are very rare, device-specific, and sparsely distributed. Consequently, we propose a novel network training algorithm based on constrained optimization to achieve a realistic backdoor injection attack in hardware. By modifying parameters uniformly across the convolutional and fully-connected layers as well as optimizing the trigger pattern together, we achieve state-of-the-art attack performance with fewer bit flips. For instance, our method on a hardware-deployed ResNet-20 model trained on CIFAR-10 achieves over 89% test accuracy and 92% attack success rate by flipping only 10 out of 2.2 million bits.



## **37. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06607v1) [paper-pdf](http://arxiv.org/pdf/2304.06607v1)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.



## **38. EGC: Image Generation and Classification via a Diffusion Energy-Based Model**

cs.CV

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.02012v3) [paper-pdf](http://arxiv.org/pdf/2304.02012v3)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.



## **39. Certified Zeroth-order Black-Box Defense with Robust UNet Denoiser**

cs.CV

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06430v1) [paper-pdf](http://arxiv.org/pdf/2304.06430v1)

**Authors**: Astha Verma, Siddhesh Bangar, A V Subramanyam, Naman Lal, Rajiv Ratn Shah, Shin'ichi Satoh

**Abstract**: Certified defense methods against adversarial perturbations have been recently investigated in the black-box setting with a zeroth-order (ZO) perspective. However, these methods suffer from high model variance with low performance on high-dimensional datasets due to the ineffective design of the denoiser and are limited in their utilization of ZO techniques. To this end, we propose a certified ZO preprocessing technique for removing adversarial perturbations from the attacked image in the black-box setting using only model queries. We propose a robust UNet denoiser (RDUNet) that ensures the robustness of black-box models trained on high-dimensional datasets. We propose a novel black-box denoised smoothing (DS) defense mechanism, ZO-RUDS, by prepending our RDUNet to the black-box model, ensuring black-box defense. We further propose ZO-AE-RUDS in which RDUNet followed by autoencoder (AE) is prepended to the black-box model. We perform extensive experiments on four classification datasets, CIFAR-10, CIFAR-10, Tiny Imagenet, STL-10, and the MNIST dataset for image reconstruction tasks. Our proposed defense methods ZO-RUDS and ZO-AE-RUDS beat SOTA with a huge margin of $35\%$ and $9\%$, for low dimensional (CIFAR-10) and with a margin of $20.61\%$ and $23.51\%$ for high-dimensional (STL-10) datasets, respectively.



## **40. How to Sign Quantum Messages**

quant-ph

22 pages

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06325v1) [paper-pdf](http://arxiv.org/pdf/2304.06325v1)

**Authors**: Mohammed Barhoush, Louis Salvail

**Abstract**: Signing quantum messages has been shown to be impossible even under computational assumptions. We show that this result can be circumvented by relying on verification keys that change with time or that are large quantum states. Correspondingly, we give two new approaches to sign quantum information. The first approach assumes quantum-secure one-way functions (QOWF) to obtain a time-dependent signature scheme where the algorithms take into account time. The keys are classical but the verification key needs to be continually updated. The second construction uses fixed quantum verification keys and achieves information-theoretic secure signatures against adversaries with bounded quantum memory i.e. in the bounded quantum storage model. Furthermore, we apply our time-dependent signatures to authenticate keys in quantum public key encryption schemes and achieve indistinguishability under chosen quantum key and ciphertext attack (qCKCA).



## **41. Multi-Glimpse Network: A Robust and Efficient Classification Architecture based on Recurrent Downsampled Attention**

cs.CV

Accepted at BMVC 2021

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2111.02018v2) [paper-pdf](http://arxiv.org/pdf/2111.02018v2)

**Authors**: Sia Huat Tan, Runpei Dong, Kaisheng Ma

**Abstract**: Most feedforward convolutional neural networks spend roughly the same efforts for each pixel. Yet human visual recognition is an interaction between eye movements and spatial attention, which we will have several glimpses of an object in different regions. Inspired by this observation, we propose an end-to-end trainable Multi-Glimpse Network (MGNet) which aims to tackle the challenges of high computation and the lack of robustness based on recurrent downsampled attention mechanism. Specifically, MGNet sequentially selects task-relevant regions of an image to focus on and then adaptively combines all collected information for the final prediction. MGNet expresses strong resistance against adversarial attacks and common corruptions with less computation. Also, MGNet is inherently more interpretable as it explicitly informs us where it focuses during each iteration. Our experiments on ImageNet100 demonstrate the potential of recurrent downsampled attention mechanisms to improve a single feedforward manner. For example, MGNet improves 4.76% accuracy on average in common corruptions with only 36.9% computational cost. Moreover, while the baseline incurs an accuracy drop to 7.6%, MGNet manages to maintain 44.2% accuracy in the same PGD attack strength with ResNet-50 backbone. Our code is available at https://github.com/siahuat0727/MGNet.



## **42. Identification of Systematic Errors of Image Classifiers on Rare Subgroups**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2303.05072v2) [paper-pdf](http://arxiv.org/pdf/2303.05072v2)

**Authors**: Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, Dan Zhang

**Abstract**: Despite excellent average-case performance of many image classifiers, their performance can substantially deteriorate on semantically coherent subgroups of the data that were under-represented in the training data. These systematic errors can impact both fairness for demographic minority groups as well as robustness and safety under domain shift. A major challenge is to identify such subgroups with subpar performance when the subgroups are not annotated and their occurrence is very rare. We leverage recent advances in text-to-image models and search in the space of textual descriptions of subgroups ("prompts") for subgroups where the target model has low performance on the prompt-conditioned synthesized data. To tackle the exponentially growing number of subgroups, we employ combinatorial testing. We denote this procedure as PromptAttack as it can be interpreted as an adversarial attack in a prompt space. We study subgroup coverage and identifiability with PromptAttack in a controlled setting and find that it identifies systematic errors with high accuracy. Thereupon, we apply PromptAttack to ImageNet classifiers and identify novel systematic errors on rare subgroups.



## **43. Optimal Detector Placement in Networked Control Systems under Cyber-attacks with Applications to Power Networks**

eess.SY

7 pages, 4 figures, accepted to IFAC 2023

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05710v1) [paper-pdf](http://arxiv.org/pdf/2304.05710v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira, Alexander Medvedev

**Abstract**: This paper proposes a game-theoretic method to address the problem of optimal detector placement in a networked control system under cyber-attacks. The networked control system is composed of interconnected agents where each agent is regulated by its local controller over unprotected communication, which leaves the system vulnerable to malicious cyber-attacks. To guarantee a given local performance, the defender optimally selects a single agent on which to place a detector at its local controller with the purpose of detecting cyber-attacks. On the other hand, an adversary optimally chooses a single agent on which to conduct a cyber-attack on its input with the aim of maximally worsening the local performance while remaining stealthy to the defender. First, we present a necessary and sufficient condition to ensure that the maximal attack impact on the local performance is bounded, which restricts the possible actions of the defender to a subset of available agents. Then, by considering the maximal attack impact on the local performance as a game payoff, we cast the problem of finding optimal actions of the defender and the adversary as a zero-sum game. Finally, with the possible action sets of the defender and the adversary, an algorithm is devoted to determining the Nash equilibria of the zero-sum game that yield the optimal detector placement. The proposed method is illustrated on an IEEE benchmark for power systems.



## **44. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP)  (Version 8); include recent progress till Apr 2023 in Version 9; 14 pages for  the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2009.04131v9) [paper-pdf](http://arxiv.org/pdf/2009.04131v9)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstract**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.



## **45. Generative Adversarial Networks-Driven Cyber Threat Intelligence Detection Framework for Securing Internet of Things**

cs.CR

The paper is accepted and will be published in the IEEE DCOSS-IoT  2023 Conference Proceedings

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05644v1) [paper-pdf](http://arxiv.org/pdf/2304.05644v1)

**Authors**: Mohamed Amine Ferrag, Djallel Hamouda, Merouane Debbah, Leandros Maglaras, Abderrahmane Lakas

**Abstract**: While the benefits of 6G-enabled Internet of Things (IoT) are numerous, providing high-speed, low-latency communication that brings new opportunities for innovation and forms the foundation for continued growth in the IoT industry, it is also important to consider the security challenges and risks associated with the technology. In this paper, we propose a two-stage intrusion detection framework for securing IoTs, which is based on two detectors. In the first stage, we propose an adversarial training approach using generative adversarial networks (GAN) to help the first detector train on robust features by supplying it with adversarial examples as validation sets. Consequently, the classifier would perform very well against adversarial attacks. Then, we propose a deep learning (DL) model for the second detector to identify intrusions. We evaluated the proposed approach's efficiency in terms of detection accuracy and robustness against adversarial attacks. Experiment results with a new cyber security dataset demonstrate the effectiveness of the proposed methodology in detecting both intrusions and persistent adversarial examples with a weighted avg of 96%, 95%, 95%, and 95% for precision, recall, f1-score, and accuracy, respectively.



## **46. Overload: Latency Attacks on Object Detection for Edge Devices**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05370v2) [paper-pdf](http://arxiv.org/pdf/2304.05370v2)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-rung Lee

**Abstract**: Nowadays, the deployment of deep learning based applications on edge devices is an essential task owing to the increasing demands on intelligent services. However, the limited computing resources on edge nodes make the models vulnerable to attacks, such that the predictions made by models are unreliable. In this paper, we investigate latency attacks on deep learning applications. Unlike common adversarial attacks for misclassification, the goal of latency attacks is to increase the inference time, which may stop applications from responding to the requests within a reasonable time. This kind of attack is ubiquitous for various applications, and we use object detection to demonstrate how such kind of attacks work. We also design a framework named Overload to generate latency attacks at scale. Our method is based on a newly formulated optimization problem and a novel technique, called spatial attention, to increase the inference time of object detection. We have conducted experiments using YOLOv5 models on Nvidia NX. The experimental results show that with latency attacks, the inference time of a single image can be increased ten times longer in reference to the normal setting. Moreover, comparing to existing methods, our attacking method is simpler and more effective.



## **47. Enhancing the Self-Universality for Transferable Targeted Attacks**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2209.03716v3) [paper-pdf](http://arxiv.org/pdf/2209.03716v3)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstract**: In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12\% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipeng-wei/Self-Universality.



## **48. On the Adversarial Inversion of Deep Biometric Representations**

cs.CV

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2304.05561v1) [paper-pdf](http://arxiv.org/pdf/2304.05561v1)

**Authors**: Gioacchino Tangari, Shreesh Keskar, Hassan Jameel Asghar, Dali Kaafar

**Abstract**: Biometric authentication service providers often claim that it is not possible to reverse-engineer a user's raw biometric sample, such as a fingerprint or a face image, from its mathematical (feature-space) representation. In this paper, we investigate this claim on the specific example of deep neural network (DNN) embeddings. Inversion of DNN embeddings has been investigated for explaining deep image representations or synthesizing normalized images. Existing studies leverage full access to all layers of the original model, as well as all possible information on the original dataset. For the biometric authentication use case, we need to investigate this under adversarial settings where an attacker has access to a feature-space representation but no direct access to the exact original dataset nor the original learned model. Instead, we assume varying degree of attacker's background knowledge about the distribution of the dataset as well as the original learned model (architecture and training process). In these cases, we show that the attacker can exploit off-the-shelf DNN models and public datasets, to mimic the behaviour of the original learned model to varying degrees of success, based only on the obtained representation and attacker's prior knowledge. We propose a two-pronged attack that first infers the original DNN by exploiting the model footprint on the embedding, and then reconstructs the raw data by using the inferred model. We show the practicality of the attack on popular DNNs trained for two prominent biometric modalities, face and fingerprint recognition. The attack can effectively infer the original recognition model (mean accuracy 83\% for faces, 86\% for fingerprints), and can craft effective biometric reconstructions that are successfully authenticated with 1-vs-1 authentication accuracy of up to 92\% for some models.



## **49. Unfooling Perturbation-Based Post Hoc Explainers**

cs.AI

Accepted to AAAI-23. See the companion blog post at  https://medium.com/@craymichael/noncompliance-in-algorithmic-audits-and-defending-auditors-5b9fbdab2615.  9 pages (not including references and supplemental)

**SubmitDate**: 2023-04-12    [abs](http://arxiv.org/abs/2205.14772v3) [paper-pdf](http://arxiv.org/pdf/2205.14772v3)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstract**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.



## **50. ADI: Adversarial Dominating Inputs in Vertical Federated Learning Systems**

cs.CR

**SubmitDate**: 2023-04-11    [abs](http://arxiv.org/abs/2201.02775v3) [paper-pdf](http://arxiv.org/pdf/2201.02775v3)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang, Wenting Zheng

**Abstract**: Vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-aware manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individuals. Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in federated learning scenarios. We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the saliency score of ``victim'' participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.



