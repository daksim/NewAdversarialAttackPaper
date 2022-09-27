# Latest Adversarial Attack Papers
**update at 2022-09-27 09:17:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. ASK: Adversarial Soft k-Nearest Neighbor Attack and Defense**

cs.LG

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2106.14300v4)

**Authors**: Ren Wang, Tianqi Chen, Philip Yao, Sijia Liu, Indika Rajapakse, Alfred Hero

**Abstracts**: K-Nearest Neighbor (kNN)-based deep learning methods have been applied to many applications due to their simplicity and geometric interpretability. However, the robustness of kNN-based classification models has not been thoroughly explored and kNN attack strategies are underdeveloped. In this paper, we propose an Adversarial Soft kNN (ASK) loss to both design more effective kNN attack strategies and to develop better defenses against them. Our ASK loss approach has two advantages. First, ASK loss can better approximate the kNN's probability of classification error than objectives proposed in previous works. Second, the ASK loss is interpretable: it preserves the mutual information between the perturbed input and the in-class-reference data. We use the ASK loss to generate a novel attack method called the ASK-Attack (ASK-Atk), which shows superior attack efficiency and accuracy degradation relative to previous kNN attacks. Based on the ASK-Atk, we then derive an ASK-\underline{Def}ense (ASK-Def) method that optimizes the worst-case training loss induced by ASK-Atk. Experiments on CIFAR-10 (ImageNet) show that (i) ASK-Atk achieves $\geq 13\%$ ($\geq 13\%$) improvement in attack success rate over previous kNN attacks, and (ii) ASK-Def outperforms the conventional adversarial training method by $\geq 6.9\%$ ($\geq 3.5\%$) in terms of robustness improvement.



## **2. Formally verified asymptotic consensus in robust networks**

cs.PL

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2202.13833v2)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.



## **3. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2206.02829v2)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstracts**: Offline reinforcement learning (RL) provides a promising direction to exploit the massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these OOD states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.



## **4. Black-Box Dissector: Towards Erasing-based Hard-Label Model Stealing Attack**

cs.CV

**SubmitDate**: 2022-09-26    [paper-pdf](http://arxiv.org/pdf/2105.00623v3)

**Authors**: Yixu Wang, Jie Li, Hong Liu, Yan Wang, Yongjian Wu, Feiyue Huang, Rongrong Ji

**Abstracts**: Previous studies have verified that the functionality of black-box models can be stolen with full probability outputs. However, under the more practical hard-label setting, we observe that existing methods suffer from catastrophic performance degradation. We argue this is due to the lack of rich information in the probability prediction and the overfitting caused by hard labels. To this end, we propose a novel hard-label model stealing method termed \emph{black-box dissector}, which consists of two erasing-based modules. One is a CAM-driven erasing strategy that is designed to increase the information capacity hidden in hard labels from the victim model. The other is a random-erasing-based self-knowledge distillation module that utilizes soft labels from the substitute model to mitigate overfitting. Extensive experiments on four widely-used datasets consistently demonstrate that our method outperforms state-of-the-art methods, with an improvement of at most $8.27\%$. We also validate the effectiveness and practical potential of our method on real-world APIs and defense methods. Furthermore, our method promotes other downstream tasks, \emph{i.e.}, transfer adversarial attacks.



## **5. Exploiting Trust for Resilient Hypothesis Testing with Malicious Robots**

cs.RO

12 pages, 4 figures, extended version of conference submission

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2209.12285v1)

**Authors**: Matthew Cavorsi, Orhan Eren Akgün, Michal Yemini, Andrea Goldsmith, Stephanie Gil

**Abstracts**: We develop a resilient binary hypothesis testing framework for decision making in adversarial multi-robot crowdsensing tasks. This framework exploits stochastic trust observations between robots to arrive at tractable, resilient decision making at a centralized Fusion Center (FC) even when i) there exist malicious robots in the network and their number may be larger than the number of legitimate robots, and ii) the FC uses one-shot noisy measurements from all robots. We derive two algorithms to achieve this. The first is the Two Stage Approach (2SA) that estimates the legitimacy of robots based on received trust observations, and provably minimizes the probability of detection error in the worst-case malicious attack. Here, the proportion of malicious robots is known but arbitrary. For the case of an unknown proportion of malicious robots, we develop the Adversarial Generalized Likelihood Ratio Test (A-GLRT) that uses both the reported robot measurements and trust observations to estimate the trustworthiness of robots, their reporting strategy, and the correct hypothesis simultaneously. We exploit special problem structure to show that this approach remains computationally tractable despite several unknown problem parameters. We deploy both algorithms in a hardware experiment where a group of robots conducts crowdsensing of traffic conditions on a mock-up road network similar in spirit to Google Maps, subject to a Sybil attack. We extract the trust observations for each robot from actual communication signals which provide statistical information on the uniqueness of the sender. We show that even when the malicious robots are in the majority, the FC can reduce the probability of detection error to 30.5% and 29% for the 2SA and the A-GLRT respectively.



## **6. Residue-Based Natural Language Adversarial Attack Detection**

cs.CL

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2204.10192v2)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Deep learning based systems are susceptible to adversarial attacks, where a small, imperceptible change at the input alters the model prediction. However, to date the majority of the approaches to detect these attacks have been designed for image processing systems. Many popular image adversarial detection approaches are able to identify adversarial examples from embedding feature spaces, whilst in the NLP domain existing state of the art detection approaches solely focus on input text features, without consideration of model embedding spaces. This work examines what differences result when porting these image designed strategies to Natural Language Processing (NLP) tasks - these detectors are found to not port over well. This is expected as NLP systems have a very different form of input: discrete and sequential in nature, rather than the continuous and fixed size inputs for images. As an equivalent model-focused NLP detection approach, this work proposes a simple sentence-embedding "residue" based detector to identify adversarial examples. On many tasks, it out-performs ported image domain detectors and recent state of the art NLP specific detectors.



## **7. SPRITZ-1.5C: Employing Deep Ensemble Learning for Improving the Security of Computer Networks against Adversarial Attacks**

cs.CR

**SubmitDate**: 2022-09-25    [paper-pdf](http://arxiv.org/pdf/2209.12195v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Erkay Savas, Mauro Conti, Yassine Mekdad

**Abstracts**: In the past few years, Convolutional Neural Networks (CNN) have demonstrated promising performance in various real-world cybersecurity applications, such as network and multimedia security. However, the underlying fragility of CNN structures poses major security problems, making them inappropriate for use in security-oriented applications including such computer networks. Protecting these architectures from adversarial attacks necessitates using security-wise architectures that are challenging to attack.   In this study, we present a novel architecture based on an ensemble classifier that combines the enhanced security of 1-Class classification (known as 1C) with the high performance of conventional 2-Class classification (known as 2C) in the absence of attacks.Our architecture is referred to as the 1.5-Class (SPRITZ-1.5C) classifier and constructed using a final dense classifier, one 2C classifier (i.e., CNNs), and two parallel 1C classifiers (i.e., auto-encoders). In our experiments, we evaluated the robustness of our proposed architecture by considering eight possible adversarial attacks in various scenarios. We performed these attacks on the 2C and SPRITZ-1.5C architectures separately. The experimental results of our study showed that the Attack Success Rate (ASR) of the I-FGSM attack against a 2C classifier trained with the N-BaIoT dataset is 0.9900. In contrast, the ASR is 0.0000 for the SPRITZ-1.5C classifier.



## **8. Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training**

cs.LG

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2202.09514v2)

**Authors**: Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao

**Abstracts**: Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL - a general-sum Stackelberg game model called RRL-Stack - to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.



## **9. RSD-GAN: Regularized Sobolev Defense GAN Against Speech-to-Text Adversarial Attacks**

cs.SD

Paper ACCEPTED FOR PUBLICATION IEEE Signal Processing Letters Journal

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2207.06858v2)

**Authors**: Mohammad Esmaeilpour, Nourhene Chaalia, Patrick Cardinal

**Abstracts**: This paper introduces a new synthesis-based defense algorithm for counteracting with a varieties of adversarial attacks developed for challenging the performance of the cutting-edge speech-to-text transcription systems. Our algorithm implements a Sobolev-based GAN and proposes a novel regularizer for effectively controlling over the functionality of the entire generative model, particularly the discriminator network during training. Our achieved results upon carrying out numerous experiments on the victim DeepSpeech, Kaldi, and Lingvo speech transcription systems corroborate the remarkable performance of our defense approach against a comprehensive range of targeted and non-targeted adversarial attacks.



## **10. Approximate better, Attack stronger: Adversarial Example Generation via Asymptotically Gaussian Mixture Distribution**

cs.LG

**SubmitDate**: 2022-09-24    [paper-pdf](http://arxiv.org/pdf/2209.11964v1)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstracts**: Strong adversarial examples are the keys to evaluating and enhancing the robustness of deep neural networks. The popular adversarial attack algorithms maximize the non-concave loss function using the gradient ascent. However, the performance of each attack is usually sensitive to, for instance, minor image transformations due to insufficient information (only one input example, few white-box source models and unknown defense strategies). Hence, the crafted adversarial examples are prone to overfit the source model, which limits their transferability to unidentified architectures. In this paper, we propose Multiple Asymptotically Normal Distribution Attacks (MultiANDA), a novel method that explicitly characterizes adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then apply the ensemble strategy on this procedure to estimate a Gaussian mixture model for a better exploration of the potential optimization space. Drawing perturbations from the learned distribution allow us to generate any number of adversarial examples for each input. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, the samples drawn from the distribution reliably maintain the transferability. Our proposed method outperforms nine state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defence models.



## **11. Faith: An Efficient Framework for Transformer Verification on GPUs**

cs.LG

Published in ATC'22

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2209.12708v1)

**Authors**: Boyuan Feng, Tianqi Tang, Yuke Wang, Zhaodong Chen, Zheng Wang, Shu Yang, Yuan Xie, Yufei Ding

**Abstracts**: Transformer verification draws increasing attention in machine learning research and industry. It formally verifies the robustness of transformers against adversarial attacks such as exchanging words in a sentence with synonyms. However, the performance of transformer verification is still not satisfactory due to bound-centric computation which is significantly different from standard neural networks. In this paper, we propose Faith, an efficient framework for transformer verification on GPUs. We first propose a semantic-aware computation graph transformation to identify semantic information such as bound computation in transformer verification. We exploit such semantic information to enable efficient kernel fusion at the computation graph level. Second, we propose a verification-specialized kernel crafter to efficiently map transformer verification to modern GPUs. This crafter exploits a set of GPU hardware supports to accelerate verification specialized operations which are usually memory-intensive. Third, we propose an expert-guided autotuning to incorporate expert knowledge on GPU backends to facilitate large search space exploration. Extensive evaluations show that Faith achieves $2.1\times$ to $3.4\times$ ($2.6\times$ on average) speedup over state-of-the-art frameworks.



## **12. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

cs.LG

Will appear in the proceedings of ESORICS 2022; 13 pages, 6 figures,  6 tables

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2106.08746v4)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Deep reinforcement learning (DRL) is vulnerable to adversarial perturbations. Adversaries can mislead the policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle, but face challenges in practice, either by being too slow to fool DRL policies in real time or by modifying past observations stored in the agent's memory. We show that Universal Adversarial Perturbations (UAP), independent of the individual inputs to which they are applied, can fool DRL policies effectively and in real time. We introduce three attack variants leveraging UAP. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster than the frame rate (60 Hz) of image capture and considerably faster than prior attacks ($\approx 1.8$ms). Our attack technique is also efficient, incurring an online computational cost of $\approx 0.027$ms. Using two tasks involving robotic movement, we confirm that our results generalize to complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We introduce an effective technique that detects all known adversarial perturbations against DRL policies, including all universal perturbations presented in this paper.



## **13. MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**

cs.LG

To appear at the Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2207.07941v2)

**Authors**: Ali Ramezani-Kebrya, Iman Tabrizian, Fartash Faghri, Petar Popovski

**Abstracts**: Implementations of SGD on distributed systems create new vulnerabilities, which can be identified and misused by one or more adversarial agents. Recently, it has been shown that well-known Byzantine-resilient gradient aggregation schemes are indeed vulnerable to informed attackers that can tailor the attacks (Fang et al., 2020; Xie et al., 2020b). We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed. Deterministic schemes can be integrated into MixTailor on the fly without introducing any additional hyperparameters. Randomization decreases the capability of a powerful adversary to tailor its attacks, while the resulting randomized aggregation scheme is still competitive in terms of performance. For both iid and non-iid settings, we establish almost sure convergence guarantees that are both stronger and more general than those available in the literature. Our empirical studies across various datasets, attacks, and settings, validate our hypothesis and show that MixTailor successfully defends when well-known Byzantine-tolerant schemes fail.



## **14. Reducing Exploitability with Population Based Training**

cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-09-23    [paper-pdf](http://arxiv.org/pdf/2208.05083v2)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.



## **15. Privacy Attacks Against Biometric Models with Fewer Samples: Incorporating the Output of Multiple Models**

cs.CV

This is a major revision of a paper titled "Inverting Biometric  Models with Fewer Samples: Incorporating the Output of Multiple Models" by  the same authors that appears at IJCB 2022

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.11020v1)

**Authors**: Sohaib Ahmad, Benjamin Fuller, Kaleel Mahmood

**Abstracts**: Authentication systems are vulnerable to model inversion attacks where an adversary is able to approximate the inverse of a target machine learning model. Biometric models are a prime candidate for this type of attack. This is because inverting a biometric model allows the attacker to produce a realistic biometric input to spoof biometric authentication systems.   One of the main constraints in conducting a successful model inversion attack is the amount of training data required. In this work, we focus on iris and facial biometric systems and propose a new technique that drastically reduces the amount of training data necessary. By leveraging the output of multiple models, we are able to conduct model inversion attacks with 1/10th the training set size of Ahmad and Fuller (IJCB 2020) for iris data and 1/1000th the training set size of Mai et al. (Pattern Analysis and Machine Intelligence 2019) for facial data. We denote our new attack technique as structured random with alignment loss. Our attacks are black-box, requiring no knowledge of the weights of the target neural network, only the dimension, and values of the output vector.   To show the versatility of the alignment loss, we apply our attack framework to the task of membership inference (Shokri et al., IEEE S&P 2017) on biometric data. For the iris, membership inference attack against classification networks improves from 52% to 62% accuracy.



## **16. In Differential Privacy, There is Truth: On Vote Leakage in Ensemble Private Learning**

cs.LG

To appear at NeurIPS 2022

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.10732v1)

**Authors**: Jiaqi Wang, Roei Schuster, Ilia Shumailov, David Lie, Nicolas Papernot

**Abstracts**: When learning from sensitive data, care must be taken to ensure that training algorithms address privacy concerns. The canonical Private Aggregation of Teacher Ensembles, or PATE, computes output labels by aggregating the predictions of a (possibly distributed) collection of teacher models via a voting mechanism. The mechanism adds noise to attain a differential privacy guarantee with respect to the teachers' training data. In this work, we observe that this use of noise, which makes PATE predictions stochastic, enables new forms of leakage of sensitive information. For a given input, our adversary exploits this stochasticity to extract high-fidelity histograms of the votes submitted by the underlying teachers. From these histograms, the adversary can learn sensitive attributes of the input such as race, gender, or age. Although this attack does not directly violate the differential privacy guarantee, it clearly violates privacy norms and expectations, and would not be possible at all without the noise inserted to obtain differential privacy. In fact, counter-intuitively, the attack becomes easier as we add more noise to provide stronger differential privacy. We hope this encourages future work to consider privacy holistically rather than treat differential privacy as a panacea.



## **17. Fair Robust Active Learning by Joint Inconsistency**

cs.LG

11 pages, 3 figures

**SubmitDate**: 2022-09-22    [paper-pdf](http://arxiv.org/pdf/2209.10729v1)

**Authors**: Tsung-Han Wu, Shang-Tse Chen, Winston H. Hsu

**Abstracts**: Fair Active Learning (FAL) utilized active learning techniques to achieve high model performance with limited data and to reach fairness between sensitive groups (e.g., genders). However, the impact of the adversarial attack, which is vital for various safety-critical machine learning applications, is not yet addressed in FAL. Observing this, we introduce a novel task, Fair Robust Active Learning (FRAL), integrating conventional FAL and adversarial robustness. FRAL requires ML models to leverage active learning techniques to jointly achieve equalized performance on benign data and equalized robustness against adversarial attacks between groups. In this new task, previous FAL methods generally face the problem of unbearable computational burden and ineffectiveness. Therefore, we develop a simple yet effective FRAL strategy by Joint INconsistency (JIN). To efficiently find samples that can boost the performance and robustness of disadvantaged groups for labeling, our method exploits the prediction inconsistency between benign and adversarial samples as well as between standard and robust models. Extensive experiments under diverse datasets and sensitive groups demonstrate that our method not only achieves fairer performance on benign samples but also obtains fairer robustness under white-box PGD attacks compared with existing active learning and FAL baselines. We are optimistic that FRAL would pave a new path for developing safe and robust ML research and applications such as facial attribute recognition in biometrics systems.



## **18. Formulating Robustness Against Unforeseen Attacks**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2204.13779v2)

**Authors**: Sihui Dai, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific "source" threat model, when can we expect robustness to generalize to a stronger unknown "target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose adversarial training with variation regularization (AT-VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that AT-VR can lead to improved generalization to unforeseen attacks during test-time compared to standard adversarial training. Additionally, we combine variation regularization with perceptual adversarial training [Laidlaw et al. 2021] to achieve state-of-the-art robustness on unforeseen attacks. Our code is publicly available at https://github.com/inspire-group/variation-regularization.



## **19. Adversarial Formal Semantics of Attack Trees and Related Problems**

cs.GT

In Proceedings GandALF 2022, arXiv:2209.09333

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10322v1)

**Authors**: Thomas Brihaye, Sophie Pinchinat, Alexandre Terefenko

**Abstracts**: Security is a subject of increasing attention in our actual society in order to protect critical resources from information disclosure, theft or damage. The informal model of attack trees introduced by Schneier, and widespread in the industry, is advocated in the 2008 NATO report to govern the evaluation of the threat in risk analysis. Attack-defense trees have since been the subject of many theoretical works addressing different formal approaches.   In 2017, M. Audinot et al. introduced a path semantics over a transition system for attack trees. Inspired by the later, we propose a two-player interpretation of the attack-tree formalism. To do so, we replace transition systems by concurrent game arenas and our associated semantics consist of strategies. We then show that the emptiness problem, known to be NP-complete for the path semantics, is now PSPACE-complete. Additionally, we show that the membership problem is coNP-complete for our two-player interpretation while it collapses to P in the path semantics.



## **20. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

cs.CR

13 pages, 7 figures, 9 tables, Poster presented at wisec'22

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2205.08426v2)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around ~60% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.



## **21. Fingerprinting Robot Movements via Acoustic Side Channel**

cs.CR

11 pages, 4 figures, 7 tables

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10240v1)

**Authors**: Ryan Shah, Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: In this paper, we present an acoustic side channel attack which makes use of smartphone microphones recording a robot in operation to exploit acoustic properties of the sound to fingerprint a robot's movements. In this work we consider the possibility of an insider adversary who is within physical proximity of a robotic system (such as a technician or robot operator), equipped with only their smartphone microphone. Through the acoustic side-channel, we demonstrate that it is indeed possible to fingerprint not only individual robot movements within 3D space, but also patterns of movements which could lead to inferring the purpose of the movements (i.e. surgical procedures which a surgical robot is undertaking) and hence, resulting in potential privacy violations. Upon evaluation, we find that individual robot movements can be fingerprinted with around 75% accuracy, decreasing slightly with more fine-grained movement meta-data such as distance and speed. Furthermore, workflows could be reconstructed with around 62% accuracy as a whole, with more complex movements such as pick-and-place or packing reconstructed with near perfect accuracy. As well as this, in some environments such as surgical settings, audio may be recorded and transmitted over VoIP, such as for education/teaching purposes or in remote telemedicine. The question here is, can the same attack be successful even when VoIP communication is employed, and how does packet loss impact the captured audio and the success of the attack? Using the same characteristics of acoustic sound for plain audio captured by the smartphone, the attack was 90% accurate in fingerprinting VoIP samples on average, 15% higher than the baseline without the VoIP codec employed. This opens up new research questions regarding anonymous communications to protect robotic systems from acoustic side channel attacks via VoIP communication networks.



## **22. Reconstructing Robot Operations via Radio-Frequency Side-Channel**

cs.CR

10 pages, 7 figures, 4 tables

**SubmitDate**: 2022-09-21    [paper-pdf](http://arxiv.org/pdf/2209.10179v1)

**Authors**: Ryan Shah, Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected teleoperated robotic systems play a key role in ensuring operational workflows are carried out with high levels of accuracy and low margins of error. In recent years, a variety of attacks have been proposed that actively target the robot itself from the cyber domain. However, little attention has been paid to the capabilities of a passive attacker. In this work, we investigate whether an insider adversary can accurately fingerprint robot movements and operational warehousing workflows via the radio frequency side channel in a stealthy manner. Using an SVM for classification, we found that an adversary can fingerprint individual robot movements with at least 96% accuracy, increasing to near perfect accuracy when reconstructing entire warehousing workflows.



## **23. Audit and Improve Robustness of Private Neural Networks on Encrypted Data**

cs.LG

10 pages, 10 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09996v1)

**Authors**: Jiaqi Xue, Lei Xu, Lin Chen, Weidong Shi, Kaidi Xu, Qian Lou

**Abstracts**: Performing neural network inference on encrypted data without decryption is one popular method to enable privacy-preserving neural networks (PNet) as a service. Compared with regular neural networks deployed for machine-learning-as-a-service, PNet requires additional encoding, e.g., quantized-precision numbers, and polynomial activation. Encrypted input also introduces novel challenges such as adversarial robustness and security. To the best of our knowledge, we are the first to study questions including (i) Whether PNet is more robust against adversarial inputs than regular neural networks? (ii) How to design a robust PNet given the encrypted input without decryption? We propose PNet-Attack to generate black-box adversarial examples that can successfully attack PNet in both target and untarget manners. The attack results show that PNet robustness against adversarial inputs needs to be improved. This is not a trivial task because the PNet model owner does not have access to the plaintext of the input values, which prevents the application of existing detection and defense methods such as input tuning, model normalization, and adversarial training. To tackle this challenge, we propose a new fast and accurate noise insertion method, called RPNet, to design Robust and Private Neural Networks. Our comprehensive experiments show that PNet-Attack reduces at least $2.5\times$ queries than prior works. We theoretically analyze our RPNet methods and demonstrate that RPNet can decrease $\sim 91.88\%$ attack success rate.



## **24. SoK: Decentralized Finance (DeFi) Attacks**

cs.CR

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.13035v2)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstracts**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents, including both attacks and accidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our open data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.



## **25. Leveraging Local Patch Differences in Multi-Object Scenes for Generative Adversarial Attacks**

cs.CV

Accepted at WACV 2023 (Round 1)

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09883v1)

**Authors**: Abhishek Aich, Shasha Li, Chengyu Song, M. Salman Asif, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury

**Abstracts**: State-of-the-art generative model-based attacks against image classifiers overwhelmingly focus on single-object (i.e., single dominant object) images. Different from such settings, we tackle a more practical problem of generating adversarial perturbations using multi-object (i.e., multiple dominant objects) images as they are representative of most real-world scenes. Our goal is to design an attack strategy that can learn from such natural scenes by leveraging the local patch differences that occur inherently in such images (e.g. difference between the local patch on the object `person' and the object `bike' in a traffic scene). Our key idea is: to misclassify an adversarial multi-object image, each local patch in the image should confuse the victim classifier. Based on this, we propose a novel generative attack (called Local Patch Difference or LPD-Attack) where a novel contrastive loss function uses the aforesaid local differences in feature space of multi-object scenes to optimize the perturbation generator. Through various experiments across diverse victim convolutional neural networks, we show that our approach outperforms baseline generative attacks with highly transferable perturbations when evaluated under different white-box and black-box settings.



## **26. Sparse Vicious Attacks on Graph Neural Networks**

cs.LG

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09688v1)

**Authors**: Giovanni Trappolini, Valentino Maiorca, Silvio Severino, Emanuele Rodolà, Fabrizio Silvestri, Gabriele Tolomei

**Abstracts**: Graph Neural Networks (GNNs) have proven to be successful in several predictive modeling tasks for graph-structured data.   Amongst those tasks, link prediction is one of the fundamental problems for many real-world applications, such as recommender systems.   However, GNNs are not immune to adversarial attacks, i.e., carefully crafted malicious examples that are designed to fool the predictive model.   In this work, we focus on a specific, white-box attack to GNN-based link prediction models, where a malicious node aims to appear in the list of recommended nodes for a given target victim.   To achieve this goal, the attacker node may also count on the cooperation of other existing peers that it directly controls, namely on the ability to inject a number of ``vicious'' nodes in the network.   Specifically, all these malicious nodes can add new edges or remove existing ones, thereby perturbing the original graph.   Thus, we propose SAVAGE, a novel framework and a method to mount this type of link prediction attacks.   SAVAGE formulates the adversary's goal as an optimization task, striking the balance between the effectiveness of the attack and the sparsity of malicious resources required.   Extensive experiments conducted on real-world and synthetic datasets demonstrate that adversarial attacks implemented through SAVAGE indeed achieve high attack success rate yet using a small amount of vicious nodes.   Finally, despite those attacks require full knowledge of the target model, we show that they are successfully transferable to other black-box methods for link prediction.



## **27. Understanding Real-world Threats to Deep Learning Models in Android Apps**

cs.CR

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09577v1)

**Authors**: Zizhuang Deng, Kai Chen, Guozhu Meng, Xiaodong Zhang, Ke Xu, Yao Cheng

**Abstracts**: Famous for its superior performance, deep learning (DL) has been popularly used within many applications, which also at the same time attracts various threats to the models. One primary threat is from adversarial attacks. Researchers have intensively studied this threat for several years and proposed dozens of approaches to create adversarial examples (AEs). But most of the approaches are only evaluated on limited models and datasets (e.g., MNIST, CIFAR-10). Thus, the effectiveness of attacking real-world DL models is not quite clear. In this paper, we perform the first systematic study of adversarial attacks on real-world DNN models and provide a real-world model dataset named RWM. Particularly, we design a suite of approaches to adapt current AE generation algorithms to the diverse real-world DL models, including automatically extracting DL models from Android apps, capturing the inputs and outputs of the DL models in apps, generating AEs and validating them by observing the apps' execution. For black-box DL models, we design a semantic-based approach to build suitable datasets and use them for training substitute models when performing transfer-based attacks. After analyzing 245 DL models collected from 62,583 real-world apps, we have a unique opportunity to understand the gap between real-world DL models and contemporary AE generation algorithms. To our surprise, the current AE generation algorithms can only directly attack 6.53% of the models. Benefiting from our approach, the success rate upgrades to 47.35%.



## **28. I-GWAS: Privacy-Preserving Interdependent Genome-Wide Association Studies**

q-bio.GN

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2208.08361v2)

**Authors**: Túlio Pascoal, Jérémie Decouchant, Antoine Boutet, Marcus Völp

**Abstracts**: Genome-wide Association Studies (GWASes) identify genomic variations that are statistically associated with a trait, such as a disease, in a group of individuals. Unfortunately, careless sharing of GWAS statistics might give rise to privacy attacks. Several works attempted to reconcile secure processing with privacy-preserving releases of GWASes. However, we highlight that these approaches remain vulnerable if GWASes utilize overlapping sets of individuals and genomic variations. In such conditions, we show that even when relying on state-of-the-art techniques for protecting releases, an adversary could reconstruct the genomic variations of up to 28.6% of participants, and that the released statistics of up to 92.3% of the genomic variations would enable membership inference attacks. We introduce I-GWAS, a novel framework that securely computes and releases the results of multiple possibly interdependent GWASes. I-GWAS continuously releases privacy-preserving and noise-free GWAS results as new genomes become available.



## **29. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

cs.CV

accepted at ECCV 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2204.00491v2)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.



## **30. GAMA: Generative Adversarial Multi-Object Scene Attacks**

cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2209.09502v1)

**Authors**: Abhishek Aich, Calvin Khang-Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstracts**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code will be made publicly available soon.



## **31. Learn2Weight: Parameter Adaptation against Similar-domain Adversarial Attacks**

cs.LG

Accepted in COLING 2022

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2205.07315v2)

**Authors**: Siddhartha Datta

**Abstracts**: Recent work in black-box adversarial attacks for NLP systems has attracted much attention. Prior black-box attacks assume that attackers can observe output labels from target models based on selected inputs. In this work, inspired by adversarial transferability, we propose a new type of black-box NLP adversarial attack that an attacker can choose a similar domain and transfer the adversarial examples to the target domain and cause poor performance in target model. Based on domain adaptation theory, we then propose a defensive strategy, called Learn2Weight, which trains to predict the weight adjustments for a target model in order to defend against an attack of similar-domain adversarial examples. Using Amazon multi-domain sentiment classification datasets, we empirically show that Learn2Weight is effective against the attack compared to standard black-box defense methods such as adversarial training and defensive distillation. This work contributes to the growing literature on machine learning safety.



## **32. Security and Privacy of Wireless Beacon Systems**

cs.CR

13 pages, 3 figures

**SubmitDate**: 2022-09-20    [paper-pdf](http://arxiv.org/pdf/2107.05868v2)

**Authors**: Aldar C-F. Chan, Raymond M. H. Chung

**Abstracts**: Bluetooth Low Energy (BLE) beacons have been increasingly used in smart city applications, such as location-based and proximity-based services, to enable Internet of Things to interact with people in vicinity or enhance context-awareness. Their widespread deployment in human-centric applications makes them an attractive target to adversaries for social or economic reasons. In fact, beacons are reportedly exposed to various security issues and privacy concerns. A characterization of attacks against beacon systems is given to help understand adversary motives, required adversarial capabilities, potential impact and possible defence mechanisms for different threats, with a view to facilitating security evaluation and protection formulation for beacon systems.



## **33. Parallel Proof-of-Work with Concrete Bounds**

cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2204.00034v2)

**Authors**: Patrik Keller, Rainer Böhme

**Abstracts**: Authorization is challenging in distributed systems that cannot rely on the identification of nodes. Proof-of-work offers an alternative gate-keeping mechanism, but its probabilistic nature is incompatible with conventional security definitions. Recent related work establishes concrete bounds for the failure probability of Bitcoin's sequential proof-of-work mechanism. We propose a family of state replication protocols using parallel proof-of-work. Our bottom-up design from an agreement sub-protocol allows us to give concrete bounds for the failure probability in adversarial synchronous networks. After the typical interval of 10 minutes, parallel proof-of-work offers two orders of magnitude more security than sequential proof-of-work. This means that state updates can be sufficiently secure to support commits after one block (i.e., after 10 minutes), removing the risk of double-spending in many applications. We offer guidance on the optimal choice of parameters for a wide range of network and attacker assumptions. Simulations show that the proposed construction is robust against violations of design assumptions.



## **34. A Transferable and Automatic Tuning of Deep Reinforcement Learning for Cost Effective Phishing Detection**

cs.CR

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09033v1)

**Authors**: Orel Lavie, Asaf Shabtai, Gilad Katz

**Abstracts**: Many challenging real-world problems require the deployment of ensembles multiple complementary learning models to reach acceptable performance levels. While effective, applying the entire ensemble to every sample is costly and often unnecessary. Deep Reinforcement Learning (DRL) offers a cost-effective alternative, where detectors are dynamically chosen based on the output of their predecessors, with their usefulness weighted against their computational cost. Despite their potential, DRL-based solutions are not widely used in this capacity, partly due to the difficulties in configuring the reward function for each new task, the unpredictable reactions of the DRL agent to changes in the data, and the inability to use common performance metrics (e.g., TPR/FPR) to guide the algorithm's performance. In this study we propose methods for fine-tuning and calibrating DRL-based policies so that they can meet multiple performance goals. Moreover, we present a method for transferring effective security policies from one dataset to another. Finally, we demonstrate that our approach is highly robust against adversarial attacks.



## **35. Encrypted Semantic Communication Using Adversarial Training for Privacy Preserving**

cs.IT

submitted to IEEE Wireless Communications Letters

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09008v1)

**Authors**: Xinlai Luo, Zhiyong Chen, Meixia Tao, Feng Yang

**Abstracts**: Semantic communication is implemented based on shared background knowledge, but the sharing mechanism risks privacy leakage. In this letter, we propose an encrypted semantic communication system (ESCS) for privacy preserving, which combines universality and confidentiality. The universality is reflected in that all network modules of the proposed ESCS are trained based on a shared database, which is suitable for large-scale deployment in practical scenarios. Meanwhile, the confidentiality is achieved by symmetric encryption. Based on the adversarial training, we design an adversarial encryption training scheme to guarantee the accuracy of semantic communication in both encrypted and unencrypted modes. Experiment results show that the proposed ESCS with the adversarial encryption training scheme can perform well regardless of whether the semantic information is encrypted. It is difficult for the attacker to reconstruct the original semantic information from the eavesdropped message.



## **36. Catoptric Light can be Dangerous: Effective Physical-World Attack by Natural Phenomenon**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.09652,  arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.11739v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have achieved great success in many tasks. Therefore, it is crucial to evaluate the robustness of advanced DNNs. The traditional methods use stickers as physical perturbations to fool the classifiers, which is difficult to achieve stealthiness and there exists printing loss. Some new types of physical attacks use light beam to perform attacks (e.g., laser, projector), whose optical patterns are artificial rather than natural. In this work, we study a new type of physical attack, called adversarial catoptric light (AdvCL), in which adversarial perturbations are generated by common natural phenomena, catoptric light, to achieve stealthy and naturalistic adversarial attacks against advanced DNNs in physical environments. Carefully designed experiments demonstrate the effectiveness of the proposed method in simulated and real-world environments. The attack success rate is 94.90% in a subset of ImageNet and 83.50% in the real-world environment. We also discuss some of AdvCL's transferability and defense strategy against this attack.



## **37. Adversarial Color Projection: A Projector-Based Physical Attack to DNNs**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2209.02430

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.09652v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Recent advances have shown that deep neural networks (DNNs) are susceptible to adversarial perturbations. Therefore, it is necessary to evaluate the robustness of advanced DNNs using adversarial attacks. However, traditional physical attacks that use stickers as perturbations are more vulnerable than recent light-based physical attacks. In this work, we propose a projector-based physical attack called adversarial color projection (AdvCP), which performs an adversarial attack by manipulating the physical parameters of the projected light. Experiments show the effectiveness of our method in both digital and physical environments. The experimental results demonstrate that the proposed method has excellent attack transferability, which endows AdvCP with effective blackbox attack. We prospect AdvCP threats to future vision-based systems and applications and propose some ideas for light-based physical attacks.



## **38. A Systematic Evaluation of Node Embedding Robustness**

cs.LG

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.08064v2)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstracts**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has significantly increased in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring strategies computed using network properties as well as node labels. We also investigate the effect of label homophily and heterophily on robustness. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We found that node classification suffers from higher performance degradation as opposed to network reconstruction, and that degree-based and label-based attacks are on average the most damaging.



## **39. Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples**

cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2106.09947v2)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli

**Abstracts**: Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations. Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner. In this work, we overcome these limitations by: (i) categorizing attack failures based on how they affect the optimization of gradient-based attacks, while also unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel indicators of failure, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.



## **40. Evaluating Machine Unlearning via Epistemic Uncertainty**

cs.LG

Rejected at ECML 2021. Even though the paper was rejected, we want to  "publish" it on arxiv, since we believe that it is nevertheless interesting  to investigate the connections between unlearning and uncertainty v2: Added  acknowledgment and code repository

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2208.10836v2)

**Authors**: Alexander Becker, Thomas Liebig

**Abstracts**: There has been a growing interest in Machine Unlearning recently, primarily due to legal requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act. Thus, multiple approaches were presented to remove the influence of specific target data points from a trained model. However, when evaluating the success of unlearning, current approaches either use adversarial attacks or compare their results to the optimal solution, which usually incorporates retraining from scratch. We argue that both ways are insufficient in practice. In this work, we present an evaluation metric for Machine Unlearning algorithms based on epistemic uncertainty. This is the first definition of a general evaluation metric for Machine Unlearning to our best knowledge.



## **41. AdvDO: Realistic Adversarial Attacks for Trajectory Prediction**

cs.LG

To appear in ECCV 2022

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.08744v1)

**Authors**: Yulong Cao, Chaowei Xiao, Anima Anandkumar, Danfei Xu, Marco Pavone

**Abstracts**: Trajectory prediction is essential for autonomous vehicles (AVs) to plan correct and safe driving behaviors. While many prior works aim to achieve higher prediction accuracy, few study the adversarial robustness of their methods. To bridge this gap, we propose to study the adversarial robustness of data-driven trajectory prediction systems. We devise an optimization-based adversarial attack framework that leverages a carefully-designed differentiable dynamic model to generate realistic adversarial trajectories. Empirically, we benchmark the adversarial robustness of state-of-the-art prediction models and show that our attack increases the prediction error for both general metrics and planning-aware metrics by more than 50% and 37%. We also show that our attack can lead an AV to drive off road or collide into other vehicles in simulation. Finally, we demonstrate how to mitigate the adversarial attacks using an adversarial training scheme.



## **42. On the Adversarial Transferability of ConvMixer Models**

cs.LG

5 pages, 5 figures, 5 tables. arXiv admin note: substantial text  overlap with arXiv:2209.02997

**SubmitDate**: 2022-09-19    [paper-pdf](http://arxiv.org/pdf/2209.08724v1)

**Authors**: Ryota Iijima, Miki Tanaka, Isao Echizen, Hitoshi Kiya

**Abstracts**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, which means AEs generated for a source model can fool another black-box model (target model) with a non-trivial probability. In this paper, we investigate the property of adversarial transferability between models including ConvMixer, which is an isotropic network, for the first time. To objectively verify the property of transferability, the robustness of models is evaluated by using a benchmark attack method called AutoAttack. In an image classification experiment, ConvMixer is confirmed to be weak to adversarial transferability.



## **43. Reinforcement learning-based optimised control for tracking of nonlinear systems with adversarial attacks**

eess.SY

Submitted for The 10th RSI International Conference on Robotics and  Mechatronics (ICRoM 2022)

**SubmitDate**: 2022-09-18    [paper-pdf](http://arxiv.org/pdf/2209.02165v2)

**Authors**: Farshad Rahimi, Sepideh Ziaei

**Abstracts**: This paper introduces a reinforcement learning-based tracking control approach for a class of nonlinear systems using neural networks. In this approach, adversarial attacks were considered both in the actuator and on the outputs. This approach incorporates a simultaneous tracking and optimization process. It is necessary to be able to solve the Hamilton-Jacobi-Bellman equation (HJB) in order to obtain optimal control input, but this is difficult due to the strong nonlinearity terms in the equation. In order to find the solution to the HJB equation, we used a reinforcement learning approach. In this online adaptive learning approach, three neural networks are simultaneously adapted: the critic neural network, the actor neural network, and the adversary neural network. Ultimately, simulation results are presented to demonstrate the effectiveness of the introduced method on a manipulator.



## **44. Distribution inference risks: Identifying and mitigating sources of leakage**

cs.CR

14 pages, 8 figures

**SubmitDate**: 2022-09-18    [paper-pdf](http://arxiv.org/pdf/2209.08541v1)

**Authors**: Valentin Hartmann, Léo Meynent, Maxime Peyrard, Dimitrios Dimitriadis, Shruti Tople, Robert West

**Abstracts**: A large body of work shows that machine learning (ML) models can leak sensitive or confidential information about their training data. Recently, leakage due to distribution inference (or property inference) attacks is gaining attention. In this attack, the goal of an adversary is to infer distributional information about the training data. So far, research on distribution inference has focused on demonstrating successful attacks, with little attention given to identifying the potential causes of the leakage and to proposing mitigations. To bridge this gap, as our main contribution, we theoretically and empirically analyze the sources of information leakage that allows an adversary to perpetrate distribution inference attacks. We identify three sources of leakage: (1) memorizing specific information about the $\mathbb{E}[Y|X]$ (expected label given the feature values) of interest to the adversary, (2) wrong inductive bias of the model, and (3) finiteness of the training data. Next, based on our analysis, we propose principled mitigation techniques against distribution inference attacks. Specifically, we demonstrate that causal learning techniques are more resilient to a particular type of distribution inference risk termed distributional membership inference than associative learning methods. And lastly, we present a formalization of distribution inference that allows for reasoning about more general adversaries than was previously possible.



## **45. pFedDef: Defending Grey-Box Attacks for Personalized Federated Learning**

cs.LG

16 pages, 5 figures (11 images if counting sub-figures separately),  longer version of paper submitted to CrossFL 2022 poster workshop, code  available at (https://github.com/tj-kim/pFedDef_v1)

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.08412v1)

**Authors**: Taejin Kim, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstracts**: Personalized federated learning allows for clients in a distributed system to train a neural network tailored to their unique local data while leveraging information at other clients. However, clients' models are vulnerable to attacks during both the training and testing phases. In this paper we address the issue of adversarial clients crafting evasion attacks at test time to deceive other clients. For example, adversaries may aim to deceive spam filters and recommendation systems trained with personalized federated learning for monetary gain. The adversarial clients have varying degrees of personalization based on the method of distributed learning, leading to a "grey-box" situation. We are the first to characterize the transferability of such internal evasion attacks for different learning methods and analyze the trade-off between model accuracy and robustness depending on the degree of personalization and similarities in client data. We introduce a defense mechanism, pFedDef, that performs personalized federated adversarial training while respecting resource limitations at clients that inhibit adversarial training. Overall, pFedDef increases relative grey-box adversarial robustness by 62% compared to federated adversarial training and performs well even under limited system resources.



## **46. Decentralization Paradox: A Study of Hegemonic and Risky ERC-20 Tokens**

cs.CR

2022 Engineering Graduate Research Symposium (EGRS)

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.08370v1)

**Authors**: Nikolay Ivanov, Qiben Yan

**Abstracts**: In this work, we explore the class of Ethereum smart contracts called the administrated ERC20 tokens. We demonstrate that these contracts are more owner-controlled and less safe than the services they try to disrupt, such as banks and centralized online payment systems. We develop a binary classifier for identification of administrated ERC20 tokens, and conduct extensive data analysis, which reveals that nearly 9 out of 10 ERC20 tokens on Ethereum are administrated, and thereby unsafe to engage with even under the assumption of trust towards their owners. We design and implement SafelyAdministrated - a Solidity abstract class that safeguards users of administrated ERC20 tokens from adversarial attacks or frivolous behavior of the tokens' owners.



## **47. Robust Online and Distributed Mean Estimation Under Adversarial Data Corruption**

cs.CR

8 pages, 5 figures, 61st IEEE Conference on Decision and Control  (CDC)

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.09624v1)

**Authors**: Tong Yao, Shreyas Sundaram

**Abstracts**: We study robust mean estimation in an online and distributed scenario in the presence of adversarial data attacks. At each time step, each agent in a network receives a potentially corrupted data point, where the data points were originally independent and identically distributed samples of a random variable. We propose online and distributed algorithms for all agents to asymptotically estimate the mean. We provide the error-bound and the convergence properties of the estimates to the true mean under our algorithms. Based on the network topology, we further evaluate each agent's trade-off in convergence rate between incorporating data from neighbors and learning with only local observations.



## **48. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2209.04554v3)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.



## **49. Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

cs.CR

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2208.02592v2)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstracts**: In recent cyber attacks, credential theft has emerged as one of the primary vectors of gaining entry into the system. Once attacker(s) have a foothold in the system, they use various techniques including token manipulation to elevate the privileges and access protected resources. This makes authentication and token based authorization a critical component for a secure and resilient cyber system. In this paper we discuss the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework as Resilient Risk based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch and sustain any cyber attack and provides much-needed strength to critical infrastructure. We also discuss the machine learning (ML) approach for the adaptive engine to accurately classify transactions and arrive at risk scores.



## **50. Robust Prototypical Few-Shot Organ Segmentation with Regularized Neural-ODEs**

cs.CV

**SubmitDate**: 2022-09-17    [paper-pdf](http://arxiv.org/pdf/2208.12428v2)

**Authors**: Prashant Pandey, Mustafa Chasmai, Tanuj Sur, Brejesh Lall

**Abstracts**: Despite the tremendous progress made by deep learning models in image semantic segmentation, they typically require large annotated examples, and increasing attention is being diverted to problem settings like Few-Shot Learning (FSL) where only a small amount of annotation is needed for generalisation to novel classes. This is especially seen in medical domains where dense pixel-level annotations are expensive to obtain. In this paper, we propose Regularized Prototypical Neural Ordinary Differential Equation (R-PNODE), a method that leverages intrinsic properties of Neural-ODEs, assisted and enhanced by additional cluster and consistency losses to perform Few-Shot Segmentation (FSS) of organs. R-PNODE constrains support and query features from the same classes to lie closer in the representation space thereby improving the performance over the existing Convolutional Neural Network (CNN) based FSS methods. We further demonstrate that while many existing Deep CNN based methods tend to be extremely vulnerable to adversarial attacks, R-PNODE exhibits increased adversarial robustness for a wide array of these attacks. We experiment with three publicly available multi-organ segmentation datasets in both in-domain and cross-domain FSS settings to demonstrate the efficacy of our method. In addition, we perform experiments with seven commonly used adversarial attacks in various settings to demonstrate R-PNODE's robustness. R-PNODE outperforms the baselines for FSS by significant margins and also shows superior performance for a wide array of attacks varying in intensity and design.



